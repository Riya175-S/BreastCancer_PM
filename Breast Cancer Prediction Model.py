#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #for arrays
import pandas as pd #for reading a csv file
import sklearn.datasets #to load data
from sklearn.model_selection import train_test_split #to split datasets into subsets that minimize the 
                                                     #potential for bias in your evaluation and validation process
from sklearn.linear_model import LogisticRegression #importing logistic regression
from sklearn.metrics import accuracy_score
#metric functions : prediction error for specific purposes : calculate the accuracy score for a set of predicted labels against the true values


# In[5]:


# loading the data from sklearn
data_frame= sklearn.datasets.load_breast_cancer()
print(data_frame)


# In[7]:


data_frame.head()


# In[ ]:


data_frame.size


# In[ ]:


data_frame.shape


# In[ ]:


# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)


# In[ ]:


# print the first 5 rows of the dataframe
data_frame.head()


# In[ ]:


# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target


# In[ ]:


# print last 5 rows of the dataframe
data_frame.tail()


# In[ ]:


# number of rows and columns in the dataset
data_frame.shape


# In[ ]:


# getting some information about the data
data_frame.info()


# In[ ]:


data_frame.isnull().sum()
#checking if there is any null values present


# In[ ]:


data_frame.isnull()


# In[ ]:


# statistical measures about the data
data_frame.describe()


# In[ ]:


# checking the distribution of Target Varibale
data_frame['label'].value_counts()


# In[ ]:


data_frame.groupby('label').mean()


# In[ ]:


X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
print(Y)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
#  X_TRAIN : includes all your independent variable used to train the model
# test size : 0.2 :: 20% to train, 80% testing data
# y train : dependent variable which needs to be predicted
# y test : this data has categorylabels against your dependent variable


# In[ ]:


print(X.shape, X_train.shape, X_test.shape)


# In[ ]:


#Applying logistic regression
model = LogisticRegression()


# In[ ]:


# training the Logistic Regression model using Training data

model.fit(X_train, Y_train)


# In[ ]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

#recall and precision
#confusion matrix


# In[ ]:


print('Accuracy on training data = ', training_data_accuracy)


# In[ ]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)


# In[ ]:


print('Accuracy on test data = ', test_data_accuracy)


# In[ ]:


input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is at maligant stage')

else:
  print('The Breast Cancer is at mild stage')


# In[ ]:


#doctor's details
#community


# In[ ]:




