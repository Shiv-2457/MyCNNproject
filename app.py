import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
MODEL_PATH = "waste_classification_model.h5"  # Ensure this file is uploaded
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['Recyclable Waste', 'Organic Waste']  # Adjust based on your training labels

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dims for model
    return image

# Streamlit UI
st.title("♻️ Waste Classification App")
st.write("Upload an image to classify whether it's **Recyclable or Organic Waste**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Convert to image format
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    class_index = np.argmax(prediction)
    
    st.write(f"### 🏷️ Predicted Class: **{class_labels[class_index]}**")
