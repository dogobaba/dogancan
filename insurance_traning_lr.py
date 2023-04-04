#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


data=pd.read_csv("desktop/heart_disease_data.csv")


# In[9]:


data.head()


# In[10]:


data.tail()


# In[13]:


data.info()


# In[14]:


data.describe().T


# In[16]:


data.isnull().sum()


# In[18]:


data["Data_Value"]


# In[22]:


data=pd.read_csv("Desktop/insurance.csv")


# In[24]:


data.info()


# In[26]:


data.tail()


# In[27]:


data.isnull().sum()


# In[29]:


data.sex=pd.Categorical(data.sex)


# In[30]:


data.region=pd.Categorical(data.region)


# In[31]:


data.smoker=pd.Categorical(data.smoker)


# In[32]:


data.info()


# In[34]:


data.describe().T


# In[35]:


import seaborn as sns


# In[36]:


data_num = data.select_dtypes(include = ["float64", "int64"])


# In[37]:


data_num.head()


# In[38]:


print("Ortalama: " + str(data_num["age"].mean()))


# In[45]:


data.charges.plot.hist()


# In[46]:


data.corr()


# In[47]:


from sklearn.linear_model import LinearRegression


# In[49]:


data.head()


# In[48]:


lr=LinearRegression()


# In[51]:


a=data.iloc[:,:6]


# In[52]:


a.head(10)


# In[53]:


b=data.iloc[:,6]


# In[55]:


b.head()


# In[56]:


a=pd.get_dummies(a,drop_first=True)


# In[57]:


a.head()


# In[58]:


lr.fit(a,b)


# In[60]:


lr.coef_


# In[63]:


pd.concat([pd.Series(a.columns),pd.Series(lr.coef_)],axis=1)


# In[64]:


lr.score(a,b)


# In[ ]:




