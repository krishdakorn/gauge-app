import os
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
from pymongo import MongoClient
from dotenv import load_dotenv
import base64

# ---------- Load environment ----------
load_dotenv()
MONGO_URI = os.environ.get("MONGO_URI")
GAUGE_DB = os.environ.get("GAUGE_DB", "gauge_db")
PORT = int(os.environ.get("PORT", 5000))

# ---------- MongoDB ----------
client = MongoClient(MONGO_URI)
db = client[GAUGE_DB]
gauge_col = db["gauge01"]

# ---------- Flask ----------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # อนุญาตทุก origin เฉพาะ dev

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ---------- YOLO ----------
model = YOLO("models/yolov8nyolov8s-cls_gauge-fe_best.pt")

CLASS_MAP = {
    "under_pressure": "NG : Lo",
    "in_pressure": "OK",
    "over_pressure": "NG : Hi"
}

# ---------- Upload API ----------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.form
    gauge_id = data.get("gauge_id", "FE0001")
    val_read = data.get("val_read", "")
    lat = data.get("lat")
    lon = data.get("lon")
    ip = data.get("ip")

    # รับรูปจาก base64
    img_base64 = data.get("image")
    if not img_base64:
        return jsonify({"error": "No image provided"}), 400

    # ตัด prefix "data:image/jpeg;base64,"
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]

    # แปลงเป็น bytes
    img_bytes = base64.b64decode(img_base64)

    # สร้าง timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{gauge_id}_{timestamp}.jpg"
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    result_path = os.path.join(RESULT_FOLDER, f"{gauge_id}_{timestamp}_result.jpg")

    # บันทึกรูปต้นฉบับ
    with open(upload_path, "wb") as f:
        f.write(img_bytes)

    # ---------- YOLO Classification Predict ----------
    results = model.predict(upload_path, imgsz=640, save=False)

    # YOLO Classification จะคืน result[0].probs
    res = results[0]
    class_id = int(res.probs.top1)
    
    # filter hidden / checkpoint classes สำหรับ mapping
    filtered_names = {i: n for i, n in model.names.items() if n != ".ipynb_checkpoints"}

    # ใช้ class_id ของ res
    class_id = int(res.probs.top1)
    class_name = filtered_names.get(class_id, "Unknown")

    # convert class → label
    val_ai = CLASS_MAP.get(class_name, "Unknown")

    # แทนที่จะใช้ res.plot() แบบเดิม
    # สร้างภาพใหม่จาก original image เพื่อไม่ให้ .ipynb_checkpoints โผล่
    img = cv2.imread(upload_path)
    cv2.putText(img, val_ai, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(result_path, img)

    # บันทึกลง MongoDB
    doc = {
        "gauge_id": gauge_id,
        "ip": ip,
        "lat": lat,
        "lon": lon,
        "timestamp": timestamp,
        "val_ai": val_ai,
        "val_read": val_read,
        "image": "/" + upload_path,
        "result_image": "/" + result_path
    }
    gauge_col.insert_one(doc)

    return jsonify({"status": "ok", "result_image": "/" + result_path})


# ---------- Health check ----------
@app.route("/health", methods=["GET"])
def health():
    try:
        client.admin.command("ping")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
