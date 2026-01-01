import streamlit as st

# Set page title and icon (favicon)
st.set_page_config(
    page_title="MNIST Digit Predictor",
    page_icon="🔢",   # You can use an emoji or a local image file like "icon.png"
    layout="centered"
)

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from PIL import Image, ImageOps

# Function to create and load a pre-trained MNIST model
@st.cache_resource
def get_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 784)).astype("float32") / 255
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=3, batch_size=128, verbose=0)
    return model

model = get_model()

# Streamlit UI
st.title("🔢 MNIST Handwritten Digit Predictor")
st.write("Upload an image of a digit (28x28 pixels) and the model will predict it.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image.resize((28, 28)))
    st.image(image, caption="Uploaded Image", width=150)
    
    img_array = np.array(image).reshape(1, 784).astype("float32") / 255
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    st.success(f"Predicted Digit: {predicted_label}")


