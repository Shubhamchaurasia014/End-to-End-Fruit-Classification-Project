#!/usr/bin/env python
# coding: utf-8

# # Fruit Classification - End to End Project

# ## Import libraries

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# ## Load the data into train and test

# In[3]:


## Train Data Generator

train_data_gen = ImageDataGenerator(rescale=1./255)
train_data = train_data_gen.flow_from_directory(r"C:\Users\Bhawna\OneDrive\Desktop\New folder\fruits-360_100x100\fruits-360\Training", 
                                               batch_size=32, target_size=(100,100), class_mode='categorical')

## Test Data Generator

test_data_gen = ImageDataGenerator(rescale=1./255)
test_data = test_data_gen.flow_from_directory(r"C:\Users\Bhawna\OneDrive\Desktop\New folder\fruits-360_100x100\fruits-360\Test",
                                             batch_size=32, target_size=(100,100), class_mode='categorical')


# ## Build the model

# In[7]:


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])


# In[8]:


model.compile( optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# In[10]:


model.fit(train_data, epochs=1, validation_data=[test_data], batch_size=64)


# In[11]:


model.save('fruit_model.h5')


# In[5]:


import json

with open('class_indices.json','w') as f:
    json.dump(train_data.class_indices, f)


# In[ ]:




