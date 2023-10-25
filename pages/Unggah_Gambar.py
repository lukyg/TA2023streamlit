import streamlit as st
import os

from datetime import datetime
from implement import draw_text_on_image, OCR
from pgadmin_connect import conn, cur

# Define the upload directory and create it if it doesn't exist
UPLOAD_PATH = "D:/TA2023streamlit/static/upload"
ROI_PATH = "D:/TA2023streamlit/static/roi"  # Update with the correct path to your ROI images
RESULT_PATH = "D:/TA2023streamlit/static/result"  # Update with the correct path to your result images

if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
if not os.path.exists(ROI_PATH):
    os.makedirs(ROI_PATH)
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

# Function to process the uploaded image and save it
def upload_and_process_image():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Get the filename and path to save the uploaded image
        filename = os.path.join(UPLOAD_PATH, uploaded_file.name)

        with open(filename, "wb") as f:
            f.write(uploaded_file.read())

        # Process the uploaded image
        result = draw_text_on_image(filename, uploaded_file.name)
        result_str = "Kosong" if result is None else str(result)

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Create a two-column layout
        col1, col2 = st.columns(2)

        # Column 1 - Processed Image
        with col1:
            processed_image_path = os.path.join(f"file://{RESULT_PATH}", uploaded_file.name)
            st.image(processed_image_path, caption="Processed Image", use_column_width=True)

            # Insert data into the database
            timestamp = datetime.now()
            cur.execute("INSERT INTO hasil_deteksi(filename, hasil_pembacaan, timestamp) VALUES (%s, %s, %s)",
                        (uploaded_file.name, result_str, timestamp))
            conn.commit()

        # Column 2 - Text Result and Additional Images
        with col2:
            st.write("Result:", result_str)

            if result is not None:
                # Display the ROI image
                roi_image_path = os.path.join(ROI_PATH, uploaded_file.name)
                st.image(roi_image_path, caption="ROI from Image", use_column_width=True)
                st.write("Text Result:", result_str)

                # Display the final result image
                st.write("Final Result:")
                result_image_path = os.path.join(RESULT_PATH, uploaded_file.name)
                st.image(result_image_path, use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    st.title("Image Upload and Processing")
    upload_and_process_image()
