import cv2
import streamlit as st
import numpy as np

# Function to capture images from a webcam
def capture_image(device_id):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        st.error("Unable to access the camera. Make sure it's connected and not in use by other applications.")
        return None
    ret, frame = cap.read()
    cap.release()
    return frame

# Streamlit app
st.title("Webcam Photo Capture")

# Webcam selection
st.sidebar.header("Select Webcam:")
device_id = st.sidebar.radio("Choose Webcam", ('Internal', 'External'))

if device_id == 'Internal':
    device_id = 0
else:
    device_id = 1

# Camera preview
st.header("Camera Preview")
camera_preview = st.empty()
if st.button("Start Preview"):
    while True:
        frame = capture_image(device_id)
        if frame is not None:
            camera_preview.image(frame, use_column_width=True)
        else:
            break
        if not st.session_state.previewing:
            break

# Debounce time
st.session_state.previewing = True
if st.button("Stop Preview"):
    st.session_state.previewing = False

# Capture image
st.header("Capture Image")
capture_button = st.button("Capture Image")

if capture_button:
    captured_frame = capture_image(device_id)
    if captured_frame is not None:
        st.image(captured_frame, use_column_width=True)
        st.write("Image captured successfully!")

st.sidebar.title("Settings")
debounce_time = st.sidebar.slider("Debounce Time (ms)", 50, 500, 100)
st.experimental_set_query_params(debounce=debounce_time)

st.write("Note: You can change the debounce time in the sidebar.")

st.sidebar.markdown('---')
st.sidebar.write("Made with ❤️ by Your Name")

# Full code URL
st.sidebar.markdown("Code available on [GitHub](https://github.com/yourusername/your-repo)")

# Streamlit settings
if 'debounce' in st.session_state:
    st.experimental_rerun()
