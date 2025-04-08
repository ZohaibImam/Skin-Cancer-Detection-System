import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Rest of your original code remains exactly the same...

# Load the trained model
model = load_model('skin_cancer_cnn.h5')

# Function to preprocess and predict the image
def predict_skin_cancer(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))  # Load Image
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make Prediction
    prediction = model.predict(img_array)
    probability = prediction[0][0]  # Probability of being Malignant
    benign_probability = 1 - probability  # Probability of being Benign
    class_label = "Malignant" if probability > 0.5 else "Benign"

    # Return the relevant probability based on the predicted class
    if class_label == "Malignant":
        return class_label, img, probability
    else:
        return class_label, img, benign_probability


# Streamlit App
st.title("Skin Cancer Detection")

st.markdown("""
    This is a skin cancer detection application. Upload an image, and the model will predict whether the skin lesion is **Malignant** or **Benign**.
""")

# File uploader
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Predict and display results
    class_label, img, probability = predict_skin_cancer(uploaded_image, model)

    # Display the result image with the prediction title
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)  # Show the uploaded image first
    st.write(f"**Prediction: {class_label}**")

    # Display the relevant probability
    if class_label == "Malignant":
        st.write(f"**Probability of Malignant: {probability * 100:.2f}%**")
    else:
        st.write(f"**Probability of Benign: {probability * 100:.2f}%**")

# Additional info and styling
st.markdown("""
    ### About the Model:
    This model uses CNN architecture for predicting whether a skin lesion is **Benign** or **Malignant** based on images of skin lesions.

    #### Features:
    - **Input**: Skin lesion images
    - **Output**: **Benign** or **Malignant** classification

    #### How to use:
    1. Upload an image of a skin lesion.
    2. The model will predict if it's **Benign** or **Malignant**.
""")