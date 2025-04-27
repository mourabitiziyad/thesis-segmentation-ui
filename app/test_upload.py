import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("Image Upload Test")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file
    try:
        # Option 1: Use OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.success("Successfully loaded image with OpenCV")
    except Exception as e:
        st.error(f"Error with OpenCV approach: {e}")
        # Reset file pointer
        uploaded_file.seek(0)
        try:
            # Option 2: Use PIL
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image (PIL)', use_column_width=True)
            st.success("Successfully loaded image with PIL")
        except Exception as e:
            st.error(f"Error with PIL approach: {e}")

    st.write("File details:")
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"File type: {uploaded_file.type}")
    st.write(f"File size: {uploaded_file.size} bytes") 