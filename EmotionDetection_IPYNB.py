#!/usr/bin/env python
# coding: utf-8

# In[3]:





# In[4]:


#Importing Libraries
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Activation
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# In[7]:


# Defining Variables and Setting path
images=[]
targets=[]

# {angry : 0, happy : 1, sad : 2,neutral : 3, surprise : 4 } - 5 Diffetent EMOTIONS
angry = r'E:\st\Emotion-Detection-Emotion-Detection-Branch\Dataset\angry'
happy = r'E:\st\Emotion-Detection-Emotion-Detection-Branch\Dataset\happy'
sad = r'E:\st\Emotion-Detection-Emotion-Detection-Branch\Dataset\sad'
neutral = r'E:\st\Emotion-Detection-Emotion-Detection-Branch\Dataset\neutral'
surprise = r'E:\st\Emotion-Detection-Emotion-Detection-Branch\Dataset\surprise'
fear = r'E:\st\Emotion-Detection-Emotion-Detection-Branch\Dataset\fear'
disgust = r'E:\st\Emotion-Detection-Emotion-Detection-Branch\Dataset\disgust'


# In[8]:


# Getting Image, Resizing, Converting It To Gray Scale, Storing in List Variables
content=os.listdir(angry)

for image in content:
    try:
        image_path=angry + '\\'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(48,48))
        images.append(resized_image)
        targets.append(0)
    except Exception as e:
        print("exception", e)


# In[9]:


# Getting Image, Resizing, Converting It To Gray Scale, Storing in List Variables
content=os.listdir(happy)

for image in content:
    try:
        image_path=happy + '\\'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(48,48))
        images.append(resized_image)
        targets.append(1)
    except Exception as e:
        print("exception", e)


# In[10]:


# Getting Image, Resizing, Converting It To Gray Scale, Storing in List Variables
content=os.listdir(sad)

for image in content:
    try:
        image_path=sad + '\\'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(48,48))
        images.append(resized_image)
        targets.append(2)
    except Exception as e:
        print("exception", e)


# In[11]:


# Getting Image, Resizing, Converting It To Gray Scale, Storing in List Variables
content=os.listdir(neutral)

for image in content:
    try:
        image_path=neutral + '\\'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(48,48))
        images.append(resized_image)
        targets.append(3)
    except Exception as e:
        print("exception", e)


# In[12]:


# Getting Image, Resizing, Converting It To Gray Scale, Storing in List Variables
content=os.listdir(surprise)

for image in content:
    try:
        image_path=surprise + '\\'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(48,48))
        images.append(resized_image)
        targets.append(4)
    except Exception as e:
        print("exception", e)


# In[13]:


# Getting Image, Resizing, Converting It To Gray Scale, Storing in List Variables
content=os.listdir(fear)

for image in content:
    try:
        image_path=fear + '\\'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(48,48))
        images.append(resized_image)
        targets.append(5)
    except Exception as e:
        print("exception", e)


# In[14]:


# Getting Image, Resizing, Converting It To Gray Scale, Storing in List Variables
content=os.listdir(disgust)

for image in content:
    try:
        image_path=disgust + '\\'+ image
        image=cv2.imread(image_path)
        image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resized_image=cv2.resize(image_grey,(48,48))
        images.append(resized_image)
        targets.append(6)
    except Exception as se:
        print("exception", e)


# In[15]:


# Normalization
images = np.array(images)/255.0
targets = np.array(targets)/255.0


# In[16]:


# Defining list variable for training
X_train, X_test, Y_train, Y_test = train_test_split(images, targets, test_size=0.3)


# In[17]:


# Checking Array Dimensions
X_train.ndim


# In[18]:


# Reshaping
X_train=X_train.reshape(X_train.shape[0],48, 48, 1)
X_test=X_test.reshape(X_test.shape[0],48,48,1)


# In[19]:


# Encoding For Handling Categorical Values
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(Y_train)
Y_test=label_encoder.fit_transform(Y_test)


# In[20]:


# Converting To Binary Class Matrix
Y_train=np_utils.to_categorical(Y_train)


# In[21]:


# Creating Sequential model and Layers
model= Sequential()

model.add(Conv2D(200, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(150, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(100, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout((0.4)))
model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(Dense(7,activation='softmax'))

# Compiling Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Saving Model For Future Use
cp = ModelCheckpoint('model-ED', verbose=0, save_best_only=True)
# Training Model
model.fit(X_train, Y_train, epochs = 25,  callbacks=[cp], validation_split=0.2)


# In[ ]:


# Importing Libraries For Loading Model
from keras.models import load_model


# In[ ]:


# Loading Model
model=load_model('model-best')


# In[ ]:


# For Face Detection
face_detect=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')


# In[ ]:


# Capturing Video
source = cv2.VideoCapture(0)

while True:
    not_to_use, image = source.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.5,5)
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+w, x:x+w]
        resized_face = cv2.resize(face_roi, (100, 100))
        normalized_Face = resized_face/255
        reshaped_face = np.reshape(normalized_Face, (1, 100, 100, 1))
        result = model.predict(reshaped_face)[0]
        # Using 5 Emotions as per definition

        if np.amax(result)==result[0]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(image,"Angry",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)
        if np.amax(result)==result[1]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(image,"happy",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,255),thickness=2)
        if np.amax(result)==result[2]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(image,"neutral",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,255,0),thickness=2)
        if np.amax(result)==result[3]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,50),2)
            cv2.putText(image,"sad",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,255,50),thickness=2)
        if np.amax(result)==result[4]:
            cv2.rectangle(image,(x,y),(x+w,y+h),(17,17,17),2)
            cv2.putText(image,"surprised",(x,y),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(17,17,17),thickness=2)

    Height=600
    Width=600
    dimension = (Width, Height)
    resized_image = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)

    cv2.imshow('Image Processing Review 3', resized_image)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
source.release()



# In[ ]:




