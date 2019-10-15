#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv('Downloads/Training.csv')

df.head()


# In[4]:


X = df.iloc[:, :-1]
y = df['prognosis']


# In[5]:


# Train, Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)


# In[6]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()


# In[7]:


rf_clf.fit(X_train, y_train)

print("Accuracy on split test: ", rf_clf.score(X_test,y_test))


# In[8]:


# Load real test data
df_test = pd.read_csv('Downloads/Testing.csv')


# In[9]:


X_acutal_test = df_test.iloc[:, :-1]
y_actual_test = df_test['prognosis']


# In[10]:


print("Accuracy on acutal test: ", rf_clf.score(X_acutal_test, y_actual_test))


# In[12]:


symptoms_dict = {}

for index, symptom in enumerate(X):
    symptoms_dict[symptom] = index


# In[13]:


symptoms_dict


# In[27]:


# slicing fisrt 10 symptoms 
import itertools
D=dict(itertools.islice(symptoms_dict.items(), 15))


# In[28]:


# Accurance of symptoms in test data (first 15)
import seaborn as sns

sns.barplot(list(D.values()), list(D.keys()))


# In[29]:


import matplotlib.pylab as plt

lists = sorted(D.items())

x, y = zip(*lists) 

plt.plot(y,x)
plt.show()


# In[12]:


input_vector = np.zeros(len(symptoms_dict))


# In[21]:


input_vector[[symptoms_dict['itching'], symptoms_dict['skin_rash'],symptoms_dict['nodal_skin_eruptions']]] = 1


# In[22]:


rf_clf.predict_proba([input_vector])


# In[23]:


rf_clf.predict([input_vector])


# In[ ]:




