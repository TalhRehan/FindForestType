import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

# Loading the model
model = pickle.load(open('rfc.pkl', 'rb'))

# Creating the web app
st.title("Forest Cover Type Prediction")
image = Image.open('img.PNG')
st.image(image, use_column_width=True)

user_input = st.text_input('Enter all cover type features')

if user_input:
    user_input = user_input.split(',')
    features = np.array([user_input], dtype=np.float64)
    prediction = model.predict(features).reshape(1, -1)
    prediction = int(prediction[0])

    cover_type_dict = {
        1: {'name': 'Spruce/Fir', 'image': 'img_1.png'},
        2: {'name': 'Lodgepole Pine', 'image': 'img_2.png'},
        3: {'name': 'Ponderosa Pine', 'image': 'img_3.png'},
        4: {'name': 'Cottonwood/Willow', 'image': 'img_4.png'},
        5: {'name': 'Aspen', 'image': 'img_5.png'},
        6: {'name': 'Douglas-fir', 'image': 'img_6.png'},
        7: {'name': 'Krummholz', 'image': 'img_7.png'}
    }

    cover_type_info = cover_type_dict.get(prediction)

    if cover_type_info is not None:
        forest_name = cover_type_info['name']
        forest_img = cover_type_info['image']

        col1, col2 = st.columns([2, 3])

        with col1:
            st.write('This is the predicted cover type:')
            st.write(forest_name, unsafe_allow_html=True)

        with col2:
            final_image = Image.open(forest_img)
            st.image(final_image, caption=forest_name, use_column_width=True)
