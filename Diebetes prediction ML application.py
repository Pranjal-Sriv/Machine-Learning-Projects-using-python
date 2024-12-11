#!/usr/bin/env python
# coding: utf-8

# ## Diebetes Prediction using Support vector machines

# ## Work flow 
# 1. Get dataset
# 2. Data preprocessing
# 3. Test train Split
# 4. Support vector machines classifier
# 5. Finding accuracy score to know if model is trained correctly
# 6. provide new data to trained support vector machine classifier
# 7. Prediction 
# 
# 

# ## Key learnings
# 1. standard scaler in sklearn
# 2. use of numpy 
# 3. use of pandas 
# 4. tail and head fns
# 5. pd.read_csv?
# 6. shape fn
# 7. value_counts() fn
# 8. groupby fn
# 9. dropping a column ; why axis=1 is required? why axis=0 is required?
# 10. What is data standardization? why it is required?
# 11. fitting and transforming in standard scaler meaning and reason
# 12. fit_transform fn
# 13. how to apply train_test_split ? why it is used?
# 14. Explain variables test_size,stratify,random_state
# 15. classifier=svm.SVC(kernel='linear') ? what is kernel
# 16. What does SVC fn does?
# 17. different model evaluation metrics and their meaning ?
# 18. what is good accuracy score? why?
# 19. optimization techniques to increase accuracy score?

# In[6]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# ## Data collection and analysis

# In[7]:


diabetes_dataset=pd.read_csv("diabetes.csv")


# In[8]:


diabetes_dataset.tail()


# In[9]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# ## study of features
# 1. pregnancies: dataset is about females
# 2. Skin thickness:calculated from fat content in tricep muscle
# 3. BMI: body mass index i.e weight/(height*height)
# 4. DiabetesPedigreeFunction : a diabetic value
# 5. Outcome: classify if patient is diabetic or not

# In[10]:


diabetes_dataset.shape


# In[11]:


# Getting statistical measures
diabetes_dataset.describe()


# In[12]:


#To check current no. of diabetic and non-diabetic patients
# 1 ---> Diabetic person
# 2 ---> Non-diabetic person

diabetes_dataset['Outcome'].value_counts()


# In[13]:


## data grouping based on 'Outcome'
diabetes_dataset.groupby('Outcome').mean()


# #Insights
# 1. people with diabetes have more glucose level compared to non-diabetic
# 2. Insulin,skin-thickness,BMI and age are higher for diabetic people

# In[14]:


#Separating data and labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']


# In[15]:


print(X)


# In[16]:


print(Y)


# ## Data Standardization
# to get data under same range

# In[17]:


scaler=StandardScaler()


# In[18]:


scaler.fit(X)


# In[19]:


standardized_data=scaler.transform(X)


# In[20]:


print(standardized_data)


# all values now are between 0 and 1 , which is good for ML model

# In[21]:


X=standardized_data
Y=diabetes_dataset['Outcome']


# In[22]:


print(X)


# ## test_train split

# In[23]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)  


# In[24]:


print(X.shape,X_train.shape,X_test.shape)


# ## Model Training

# In[25]:


classifier=svm.SVC(kernel='linear')


# In[26]:


# Training the support vector machine classifier
classifier.fit(X_train,Y_train)


# ## Model Evaluation
# 
# 

# In[27]:


#Accuracy score on the training data
X_train_prediction=classifier.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[28]:


print("Accuracy score of the training data:",training_data_accuracy)


# out of 100 predictions , model prediction is correct 79 times

# In[30]:


#Accuracy score on the test data
X_test_prediction=classifier.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[31]:


print("Accuracy score of the test data:",test_data_accuracy)


# In overfitting, accuracy of training data will be very high
# and accuracy of test data will be very low

# ## Making a predictive system

# In[38]:


input_data=(6,148,72,35,0,33.6,0.627,50)
#changing the input data to numpy array
input_data_as_numpy_array=np.array(input_data)

#reshape the array as we are predicting for one instance
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
std_data=scaler.transform(input_data_reshape)
print(std_data)

prediction=classifier.predict(std_data)
print(prediction)

if(prediction==[0]):
    print("patient is Non-diabetic")
else:
    print("patient is diabetic")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




