from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

# load model sekali saat startup
model = YOLO("best.pt")
model.to("cpu")

@app.route("/", methods=["GET"])
def health():
    return "API ML OK"

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    file = request.files['file']

    try:
        img = Image.open(io.BytesIO(file.read()))
        results = model.predict(img)

        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            detections.append({
                'label': label,
                'confidence': round(conf, 2)
            })

        return jsonify({'detections': detections})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
