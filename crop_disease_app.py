"""
🌿 AI-Powered Crop Disease Detection System

This project uses deep learning (MobileNetV2) to detect plant diseases from leaf images.
Farmers can upload a photo and receive instant feedback on the disease type and possible remedies.

## 📦 How to Run

1. Install dependencies:
    pip install -r requirements.txt

2. Train the model (optional, pre-trained model recommended):
    python crop_disease_app.py --train

3. Run the app:
    streamlit run crop_disease_app.py

## 📸 Example Prediction
Upload a crop leaf image to get the disease prediction using MobileNetV2.

"""

# === Requirements ===
# tensorflow>=2.9.0
# numpy
# pillow
# streamlit

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import streamlit as st
import argparse

# Define class names (update as per your dataset)
CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Healthy']

MODEL_PATH = 'crop_disease_model.h5'

def train_model(train_dir='dataset/train', val_dir='dataset/val', model_output=MODEL_PATH):
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), class_mode='categorical')
    val_gen = datagen.flow_from_directory(val_dir, target_size=(224, 224), class_mode='categorical')

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save(model_output)
    print(f"Model saved to {model_output}")

def predict(image, model):
    image = image.resize((224, 224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return CLASS_NAMES[np.argmax(predictions)], np.max(predictions)

def run_app():
    st.title("🌾 AI Crop Disease Detection")
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Please train the model first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    uploaded_file = st.file_uploader("Upload leaf image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)
        label, confidence = predict(image, model)
        st.success(f"Prediction: {label} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if args.train:
        train_model()
    else:
        run_app()
