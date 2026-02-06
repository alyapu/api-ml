from flask import Flask, request, jsonify
from ultralytics import YOLO
import io
from PIL import Image

# Inisialisasi Flask
app = Flask(__name__)

# Load model YOLOv8
model = YOLO("best.pt")  # pastikan file best.pt ada di folder yang sama

@app.route('/predict', methods=['POST'])
def predict():
    # Pastikan file dikirim
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    file = request.files['file']

    try:
        # Baca gambar
        img = Image.open(io.BytesIO(file.read()))

        # Jalankan prediksi YOLO
        results = model.predict(img)

        # Ambil hasil deteksi
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


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

