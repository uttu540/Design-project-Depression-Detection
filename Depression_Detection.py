#!/usr/bin/env python
# coding: utf-8

# In[111]:


import pandas as pd
import numpy as np
from sklearn import preprocessing,model_selection
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,callbacks
from tensorflow.keras.callbacks import EarlyStopping
from google.colab import files
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
import joblib


# In[37]:


data_to_load = files.upload()


# In[76]:


import io
df = pd.read_csv(io.BytesIO(data_to_load['depression_happiness.csv']))
df


# In[77]:


from numpy.random import seed
# setting the seed
seed(0)
tf.random.set_seed(0)


# In[78]:


df.shape


# In[79]:


df.isnull().sum()


# In[80]:


#Import library:
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
#New variable for outlet
df['Feeling'] = le.fit_transform(df['How are you feeling right now?'])
df['Genders'] = le.fit_transform(df['Gender'])
df['Relation Status'] = le.fit_transform(df['Relationship status'])
df['Pressure Study'] = le.fit_transform(df['Are you feeling pressure in your study or work right now?'])
df['Financial Status'] = le.fit_transform(df['Are you happy with your financial state?'])
df['Academic Result'] = le.fit_transform(df['Are you satisfied with your academic result?'])
df['Living Place'] = le.fit_transform(df['Are you happy with your living place?'])
df['Support'] = le.fit_transform(df['Who supports you when you are not succeeding in your academic life?'])
df['Social Media'] = le.fit_transform(df['Have you used any social media within the last 6 hours?'])
df['Meal'] = le.fit_transform(df['Are you satisfied with your meal today?'])
df['Sick'] = le.fit_transform(df['Are you feeling sick/health issues today?'])
df['Hobby'] = le.fit_transform(df['Have you done any recreational activity (sports, gaming, hobby etc.) today?'])
var_mod = ['Feeling','Gender','Age','Relation Status','Financial Status','Pressure Study','Academic Result','Living Place','Support','Social Media','Meal','Sick','Hobby','How long did you sleep last night?(in hours)','Depressed']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])


# In[81]:


df.columns


# In[82]:


del df['Timestamp']
del df['Which year are you in?']
del df['How are you feeling right now?']
del df['On a scale of 1-100, how would you express this feeling?']
del df['Gender']
del df['Your location ?']
del df['Relationship status']
del df['Are you happy with your financial state?']
del df['How much have you succeeded to cope up with the environment of your educational institution?']
del df['Understanding with your family members?']
del df['Are you feeling pressure in your study or work right now?']
del df['Are you satisfied with your academic result?']
del df['Are you happy with your living place?']
del df['Who supports you when you are not succeeding in your academic life?']
del df['Have you used any social media within the last 6 hours?']
del df['Do you have inferiority complex? ']
del df['Are you satisfied with your meal today?']
del df['Are you feeling sick/health issues today?']
del df['Have you done any recreational activity (sports, gaming, hobby etc.) today?']
del df['Age']
del df['Feeling']
del df['Genders']


# In[83]:


df


# In[84]:


X = df.drop('Depressed', axis = 1)


# In[85]:


Y = df['Depressed']


# In[106]:


df.columns


# In[87]:


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size = 0.2)


# In[88]:


print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# In[89]:


from collections import Counter
print(Counter(Y_train))


# In[94]:


model=keras.Sequential([
    layers.Dense(15,activation='relu',input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(11,activation='relu'),
    layers.Dense(1),])


# In[95]:


Early_Stopping = callbacks.EarlyStopping(min_delta = 0.001, patience = 20, restore_best_weights = True)


# In[96]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy')


# In[97]:


history = model.fit(X_train,Y_train,
                  validation_data=(X_test,Y_test),batch_size=256,
                  epochs=500,callbacks=[Early_Stopping],verbose=2)


# In[98]:


accuracy=model.evaluate(X_train,Y_train)


# In[99]:


print(accuracy*100)


# In[58]:


history_df=pd.DataFrame(history.history)


# In[59]:


history_df.columns


# In[60]:


history_df.loc[:,'loss'].plot()


# In[100]:


pd.set_option('display.max_rows', None)
df['Depressed']


# In[66]:


X.columns[0]


# In[101]:


model.predict_classes(X)


# In[ ]:


from keras.models import load_model

model.save("network.h5")
loaded_model = load_model("network.h5")

accuracy = loaded_model.predict_classes(X)


# In[103]:


files.download("network.h5")


# In[ ]:




