#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json


# In[4]:


model = tf.keras.models.load_model(r"C:\Users\Bhawna\OneDrive\Desktop\New folder\fruit_model.h5")

st.title('Fruit Classification App')
uploaded_file = st.file_uploader('Upload a fruit image', type=['png','jpg','jpeg'])

with open('class_indices.json','r') as f:
    class_map = json.load(f)
inv_map = {v: k for k, v in class_map.items()}

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((100,100))
    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1,100,100,3)

    prediction = model.predict(img_array)
    fruit_class = np.argmax(prediction)

    st.image(image, caption='Uploaded Fruit', use_container_width=False)
    st.write(f'Predicted Fruit Class: {inv_map[fruit_class]}')


# In[ ]:




