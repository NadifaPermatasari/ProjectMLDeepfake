from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load model (pastikan file h5 ada di folder yang sama)
model = load_model("xception_deepfake_image.h5")

def detect_deepfake(image_file):
    img = Image.open(image_file).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)  # Normalisasi
    prediction = model.predict(img_array)[0][0]
    return {
        "deepfake": bool(prediction > 0.5),
        "confidence": float(prediction)
    }
