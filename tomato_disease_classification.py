# import libs
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras import models, layers

MODEL = tf.keras.models.load_model("model.h5")
CLASS_NAMES = ['Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_healthy']





def main():

    st.title('Tomato Disease Classification App')

    menu = ["Predict", "About"]
    choice = st.sidebar.selectbox("Select Activity", menu)


    if choice == 'Predict':
        st.subheader("Disease classification")

        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)

            img_batch = np.expand_dims(our_image, 0)
            predictions = MODEL.predict(img_batch)

            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            st.write("Predicted Disease :", predicted_class)
            st.write("Confidence level :", confidence)
            return {
                'class': predicted_class,
                'confidence': float(confidence)
            }

    elif choice == 'About':
        st.subheader("About")
        # header
        st.header("This is an application to help farmers distinguish between healthy and diseased tomato leaves.")

if __name__ is '__main__':
        main()