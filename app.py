import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import warnings
import cv2
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
warnings.filterwarnings("ignore")



st.set_page_config(
    page_title="License Plate Letters and Digit Classifier",
    page_icon = ":car:",
    initial_sidebar_state = 'auto'
)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with st.sidebar :
    st.image('images/Logo.png')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About the Dataset", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            return key
        
def size_regulator(path, image_column='path', target_size=(100, 100)):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    image = cv2.resize(image, target_size)
    return image

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('Model/Plate_Letter_Digit_Classifier.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

# Options : Home
if options == "Home" :
   st.write("# Algorex PH")
   st.image('images/Meer.jpg')
   st.write("Pinoy AI Engineer and Competitive Programmer")
   st.text("Connect with me via Linkedin : https://www.linkedin.com/in/algorexph/")
   st.text("Kaggle Account : https://www.kaggle.com/daniellebagaforomeer")
   st.title("Shikaniah Castillo")
   st.image('images/Castillo.png')
   st.write("\n")
   keep_me = '''
with st.sidebar:
        st.title("Created by :")
        st.image('images/Logo.png')
        

'''

# Options : About the Dataset
elif options == "About the Dataset" :
    st.write("""# About the Dataset : License Plate Digits Classification Dataset""")
    st.write("### Context")
    st.write("Nowadays there are a lot of applications of Machine and Deep learning. And one of the interesting applications is License Plate Detection & Recognition. This availability of this dataset can be good reason to practice your Computer Vision skills.")
    st.write("### About")
    st.write("This is an image dataset of Belgian car license plates. This dataset already comes with well-annotated XML files. The dataset contains more than 1000 images for each digit and character/letter.")
    st.write("### Where to Use?")
    st.write("This dataset can be used to classify or detect license plates on image or video frames. Images are in good quality so you might need some more computational power for such kind of tasks.")
    st.write("### Link : https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset")

# Options : Model
elif options == "Model" :
    st.write("""# Plate Letter Digit Classifier""")

    file = st.file_uploader("", type=["jpg", "png"])


    if file is None:
       st.text("Please upload an image file")
    else:
       image = Image.open(file)
       st.image(image, use_column_width=True)
       image = image.save("temp.jpg")
       image_resized = size_regulator("temp.jpg")
       image_resized = image_resized.reshape(-1, 100, 100, 1)
       prediction = model.predict(image_resized)
       #predictions = import_and_predict(image, model)

       class_names = {0 : '0', 1 : '1', 2 : '2', 3 : '3', 4 : '4', 5 : '5', 6 : '6', 7 : '7', 8 : '8', 9 : '9', 10 : 'A', 11 : 'B', 12 : 'C', 13 : 'D', 14 : 'E', 15 : 'F', 16 : 'G', 17 : 'H', 18 : 'I', 19 : 'J', 20 : 'K', 21 : 'L', 22 : 'M', 23 : 'N', 24 : 'P', 25 : 'Q', 26 : 'R', 27 : 'S',  28 : 'T', 29 : 'U', 30 : 'V', 31 : 'W', 32 : 'X', 33 : 'Y', 34 : 'Z'}

       predicted_class_index = np.argmax(prediction)
       predicted_class_name = class_names[predicted_class_index]

       string = "Detected Character is : " + predicted_class_name
       st.markdown('Answer')
       st.info(string)
