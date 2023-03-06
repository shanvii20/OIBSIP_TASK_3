#!/usr/bin/env python
# coding: utf-8

# # Email Spam Detection with Machine Learning

# #### 1. Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# #### 2. Importing dataset

# In[2]:


em=pd.read_csv('mail_data.csv')
#print few rows
em.head()


# #### 3. Analyzing dataset

# In[3]:


em.describe()


# In[4]:


# checking for null values if present
em.isnull().sum()


# In[5]:


# checking no of values in rows and columns
em.shape


# In[6]:


# Label Encoder
# label spam mails as 0 and ham mail as 1
em.loc[em['Category']== "spam",'Category']=0
em.loc[em['Category']=='ham','Category']=1
em.head()


# In[7]:


# seprating training and test data
 
X = em['Message']
Y = em['Category']


# In[8]:


print(X)


# In[9]:


print(Y)


# #### 4. Dividing the dataset for train and test data

# In[10]:


# train 80
#test 20
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size=0.20, random_state=3)


# In[11]:


# printing dimensions 
print(X.shape)
print(X_train.shape)
print(X_test.shape)


# #### 5. Feature Extraction 

# In[12]:


# convert text data to numerical values

feature_extraction= TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[13]:


print(X_train)


# In[14]:


print(X_train_features)


# #### 6. Training the model

# In[15]:


# logistic regression
model= LogisticRegression()
model.fit(X_train_features,Y_train)


# #### 7. Evaluating Model

# In[16]:


# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)


# In[17]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)


# #### 8. Building predictive system 

# In[19]:


input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')
