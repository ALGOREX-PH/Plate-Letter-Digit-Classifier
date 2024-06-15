import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import warnings
warnings.filterwarnings("ignore")



st.set_page_config(
    page_title="Plate Letter Digit Classifier",
    page_icon = ":car:",
    initial_sidebar_state = 'auto'
)

with st.sidebar:
        st.title("Created by :")
        st.image('images/Logo.png')
        st.title("Algorex PH")
        st.image('images/Meer.jpg')
        st.subheader("Pinoy AI Engineer and Competitive Programmer")
        st.text("Connect with me via Linkedin : https://www.linkedin.com/in/algorexph/")
        st.text("Kaggle Account : https://www.kaggle.com/daniellebagaforomeer")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            return key

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('Model/Plate_Letter_Digit_Classifier.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()



st.write("""
         # Plate Letter Digit Classifier
         """
         )

file = st.file_uploader("", type=["jpg", "png"])


def import_and_predict(image_data, model):
    size = (100, 100)    
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)

    class_names = {0 : '0', 1 : '1', 2 : '2', 3 : '3', 4 : '4', 5 : '5', 6 : '6', 7 : '7', 8 : '8', 9 : '9', 10 : 'A', 11 : 'B', 12 : 'C', 13 : 'D', 14 : 'E', 15 : 'F', 16 : 'G', 17 : 'H', 18 : 'I', 19 : 'J', 20 : 'K', 21 : 'L', 22 : 'M', 23 : 'N', 24 : 'P', 25 : 'Q', 26 : 'R', 27 : 'S',  28 : 'T', 29 : 'U', 30 : 'V', 31 : 'W', 32 : 'X', 33 : 'Y', 34 : 'Z'}

    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]

    string = "Detected Character is : " + predicted_class_name
    st.markdown('Answer')
    st.info(string)