import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Smart Image Enhancement and Artistic Filter Application")

# Upload gambar
uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Pilihan filter
    filter_choice = st.selectbox("Choose a filter", ["None", "Grayscale", "Sepia", "Negative", "Cartoon"])

    # Slider untuk Brightness & Contrast
    brightness = st.slider("Brightness", -100, 100, 0)
    contrast = st.slider("Contrast", 0.5, 3.0, 1.0)

    # Apply filter
    def apply_filter(img, filter_name):
        output = img.copy()
        if filter_name == "Grayscale":
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
        elif filter_name == "Sepia":
            kernel = np.array([[0.272,0.534,0.131],
                               [0.349,0.686,0.168],
                               [0.393,0.769,0.189]])
            output = cv2.transform(output, kernel)
            output = np.clip(output, 0, 255)
        elif filter_name == "Negative":
            output = cv2.bitwise_not(output)
        elif filter_name == "Cartoon":
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(output, 9, 250, 250)
            output = cv2.bitwise_and(color, color, mask=edges)
        return output

    # Apply brightness & contrast
    def adjust_brightness_contrast(img, brightness=0, contrast=1.0):
        output = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        return output

    filtered_img = apply_filter(img_bgr, filter_choice)
    final_img = adjust_brightness_contrast(filtered_img, brightness, contrast)
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    st.subheader("Filtered Image")
    st.image(final_img_rgb, use_column_width=True)

    # Save hasil
    if st.button("Download Image"):
        final_image_pil = Image.fromarray(final_img_rgb)
        final_image_pil.save("filtered_image.png")
        st.success("Image saved as filtered_image.png")
