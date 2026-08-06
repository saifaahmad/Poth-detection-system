import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

IMG_SIZE = (224, 224)

@st.cache_resource
def load_pothole_model():
    # Load the exported SavedModel folder
    model = tf.saved_model.load("pothole_model")
    # Get the serving function
    infer = model.signatures["serving_default"]
    return infer

infer_fn = load_pothole_model()

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)
    return arr

st.title("Pothole Detection Using Deep Learning")
st.write("Upload a road image and the model will predict whether it contains a pothole or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(image)

        # Convert to tensor and call SavedModel
        out = infer_fn(tf.constant(x))

        # Get the first output tensor from the dict
        pred_tensor = list(out.values())[0]
        pred = float(pred_tensor.numpy()[0][0])

        # assuming class 1 = pothole, class 0 = normal
        if pred >= 0.5:
            label = "POTHOLE DETECTED"
        else:
            label = "NORMAL ROAD"

        st.markdown(f"### Result: **{label}**")
        st.write(f"Prediction score: {pred:.3f}")
