#!/usr/bin/env python
# coding: utf-8

# # Download dataset

# # Load the dataset

# In[2]:


import numpy as np
import pandas as pd
df = pd.read_csv("Churn_Modelling.csv")


# # Perform Below Visualizations.

# ### ● Univariate Analysis

# In[3]:


import seaborn as sns
sns.histplot(df.EstimatedSalary,kde=True)


# ### ● Bi - Variate Analysis

# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(df.Balance,df.EstimatedSalary)
plt.ylim(0,15000)


# ### ● Multi - Variate Analysis

# In[11]:


import seaborn as sns
df=pd.read_csv("Churn_Modelling.csv")
sns.pairplot(df)


# #  Perform descriptive statistics on the dataset

# In[12]:


df=pd.read_csv("Churn_Modelling.csv")
df.describe(include='all')


# #  Handle the Missing values.

# In[13]:


from ast import increment_lineno
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
df=pd.read_csv("Churn_Modelling.csv")
df.head()


# #  Find the outliers and replace the outliers

# In[14]:


import pandas as pd
import matplotlib
from matplotlib import pyplot as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(10,6)
df=pd.read_csv("Churn_Modelling.csv")
df.sample(5)


# # Check for Categorical columns and perform encoding.

# In[15]:


df=pd.read_csv("Churn_Modelling.csv")
df.columns
import pandas as pd
import numpy as np
headers=['RowNumber','CustomerID','Surname','CreditScore','Geography',
 'Gender','Age','Tenure','Balance','NumofProducts','HasCard'
 'IsActiveMember','EstimatedSalary','Exited']
import seaborn as sns
df.head()


# # Split the data into dependent and independent variables

# In[16]:


x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1]._values
print(y)


# # Scale the independent variables

# In[17]:


import seaborn as sns
df=pd.read_csv("Churn_Modelling.csv")
dff=df[['Balance','Age']]
sns.heatmap(dff.corr(), annot=True)
sns.set(rc={'figure.figsize':(40,40)})


# # Split the data into training and testing

# In[18]:


from scipy.sparse.construct import random
x=df.iloc[:, 1:2].values
y=df.iloc[:,2].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print('Row count of x_train table'+'-'+str(f"{len(x_train):,}"))
print('Row count of y_train table'+'-'+str(f"{len(y_train):,}"))
print('Row count of x_test table'+'-'+str(f"{len(x_test):,}"))
print('Row count of y_test table'+'-'+str(f"{len(y_test):,}"))


# In[ ]:




