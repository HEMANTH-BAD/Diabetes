#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[13]:


diabetes_dataset = pd.read_csv("C:\Hemanth\diabetes.csv") 


# In[ ]:





# In[12]:


diabetes_dataset.head()


# In[14]:


diabetes_dataset.shape


# In[15]:


diabetes_dataset.describe()


# In[16]:


diabetes_dataset['Outcome'].value_counts()


# In[18]:


diabetes_dataset.groupby('Outcome').mean()


# In[19]:


X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[22]:


print(X)


# In[23]:


print(Y)


# In[24]:


scaler = StandardScaler()


# In[25]:


scaler.fit(X)


# In[26]:


StandardScaler(copy=True, with_mean=True, with_std=True)


# In[27]:


standardized_data = scaler.transform(X)


# In[28]:


print(standardized_data)


# In[29]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[30]:


print(X)
print(Y)


# In[31]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[32]:


print(X.shape, X_train.shape, X_test.shape)


# In[33]:


classifier = svm.SVC(kernel='linear')


# In[47]:


classifier.fit(X_train, Y_train)


# In[39]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[40]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[41]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[50]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[55]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)


# In[56]:


if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




