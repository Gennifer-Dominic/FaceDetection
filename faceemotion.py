import streamlit as st
import cv2
from keras.models import model_from_json
from PIL import Image
import numpy as np
from keras_preprocessing.image import load_img
import tempfile
import  os
from numpy import asarray

json_file=open("emotiondetection.json","r")
model_json=json_file.read()
json_file.close()
model=model_from_json(model_json)
model.load_weights("emotiondetection.h5")


label=["angry","disgust","fear","happy","neutral","sad","surprise"]

def extract_one_img(image):
    print(f"image:{image}")
    img=load_img(image, color_mode='grayscale')
    feature=np.array(img)
    feature=feature.reshape(1,48,48,1)
    return feature/255.0

def newextract_one_img(image):
    #img = Image.open(image)
    feature=asarray(image)
    feature=feature.reshape(1,48,48,1)
    return feature/255.0

st.title("Emotion Detection from Uploaded Images")

def is_image_valid(uploaded_file):
    try:
        img = Image.open(uploaded_file)
        img.verify()  # Verifies if it is a valid image
        return True
    except:
        return False

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file and is_image_valid(uploaded_file):
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
else:
    st.error("Please upload a valid image file (jpg, png, jpeg).")


if st.button("Predict"):
    image_final=[]
    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        # Convert to numpy array
        image_np = np.array(image, dtype=np.float32)
        # Resize to 48x48
        image_resized = cv2.resize(image_np, (48, 48))
        # Normalize (optional)
        image_resized = image_resized / 255.0  # Normalization to [0,1]
        # Reshape to (1,48,48,1)
        image_final = image_resized.reshape((1, 48, 48, 1))
        print(f"image_resized: {image_resized} : {image_final}")

    pred=model.predict(image_final)
    preg_label=label[pred.argmax()]
    print("model prediction : ",preg_label)
    st.write(f"Predicted Emotion: {preg_label}")
