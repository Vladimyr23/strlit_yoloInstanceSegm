import streamlit as st
import cv2
import numpy as np
import os
import shutil

import torch
from ultralytics import YOLO


# function to clean the folder
def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Function to perform prediction (Replace with your actual prediction logic)
def perform_prediction(image):
    model = YOLO('models/best.pt')
    results = []
    # clean the prediction destination folder
    clean_folder("predict/")
    with torch.no_grad():
        results = model.predict(image, conf=0.4, iou=0.08, save=True, project="predict", name="image", device='cpu')

    return results

def main():
    st.title("Image Upload and Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file as bytes
        image_bytes = uploaded_file.read()
        # Convert the bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode numpy array to OpenCV image format
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Display the image
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            prediction = perform_prediction(image)
            # File path for the image
            image_path = "predict/image/image0.jpg"
            try:
                # Read the image using OpenCV
                image = cv2.imread(image_path)
                if image is not None:
                    # Convert the image from BGR to RGB format
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Display the image
                    st.image(image_rgb, caption="Image", use_column_width=True)
                else:
                    st.error("Failed to read the image. Please check the file path.")
            except Exception as e:
                st.error(f"Error: {e}")

            st.success(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
