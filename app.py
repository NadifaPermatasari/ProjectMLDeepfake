import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="xception_deepfake_image.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fungsi deteksi
def detect_deepfake(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    return "🟥 Deepfake" if output[0][0] > 0.5 else "🟩 Real"

# Streamlit UI
st.set_page_config(page_title="Deepfake Detector")
st.title("🕵️‍♂️ Deepfake Image Detector")
uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    result = detect_deepfake(image)
    st.subheader("Result:")
    st.success(result)
