import streamlit as st
from detect import detect_deepfake
from PIL import Image

st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🕵️ Deepfake Detector")
st.write("Unggah gambar wajah dan kami akan mendeteksi apakah itu deepfake atau tidak.")

uploaded_file = st.file_uploader("Upload Gambar Wajah", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Diunggah", use_column_width=True)

    with st.spinner("Mendeteksi..."):
        result = detect_deepfake(uploaded_file)

    st.markdown("---")
    st.subheader("🔍 Hasil Deteksi")
    st.write("**Apakah Deepfake?**", "✅ Tidak" if not result["deepfake"] else "🚨 Ya!")
    st.write(f"**Tingkat Keyakinan:** {result['confidence']*100:.2f}%")
