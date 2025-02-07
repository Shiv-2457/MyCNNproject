import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model("wasteclassification.keras")

# Define class labels
labels = ["Recyclable Waste", "Organic Waste"]

# Streamlit UI
st.set_page_config(page_title="Waste Classification App", layout="wide")

st.sidebar.info(
    "**About**\n\n"
    "- This project uses a **CNN model** to classify waste into **Organic** or **Recyclable**.\n"
    "- **Dataset Provider:** Techsash (Kaggle)\n"
    "- **Model Algorithm:** Skills4Future\n"
    "- **Future Improvements:** Classification based on **plastic types, recyclables, etc.**\n"
)

st.title("♻️ Waste Classification App")

uploaded_file = st.file_uploader("📤 Upload an image for classification", type=["jpg", "png", "jpeg"])
img_url = st.text_input("🔗 Or enter an Image URL for Classification:")

if uploaded_file or img_url:
    try:
        if uploaded_file:
            img = Image.open(uploaded_file)
        else:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))

        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess Image
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        confidence = prediction[0][class_idx] * 100

        # Display Result
        st.subheader("✅ Prediction: " + labels[class_idx])
        st.success(f"🌱 This waste is **{labels[class_idx]}!**")
        st.write(f"📊 **Confidence Scores:**\n- **{labels[0]}:** {prediction[0][0] * 100:.2f}%\n- **{labels[1]}:** {prediction[0][1] * 100:.2f}%")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

