#!/usr/bin/env python
# coding: utf-8

# In[65]:



import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

from numpy.random import seed

seed(11111)
df = pd.read_excel("pv.xlsx")
df.set_index("contract_name")

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.3, random_state=25)

print(f"No. of training examples: {train.shape[0]}")
print(f"No. of testing examples: {test.shape[0]}")


# In[27]:


# Putting on index to each dataset before split it
train = train.set_index("contract_name")
test = test.set_index("contract_name")

# dataframe 
df = pd.concat([train, test], axis=0, sort=False)
df


# In[26]:


df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df


# In[130]:


df.info()


# In[131]:


df.isna().sum()


# In[132]:


columns = ['jv', 'business_unit','country','value', 'duration']

plt.figure(figsize=(16, 14))
sns.set(font_scale= 1.2)
sns.set_style('ticks')

for i, feature in enumerate(columns):
    plt.subplot(3, 3, i+1)
    sns.countplot(data=df, x=feature, hue='contract_type', palette='Paired')
    
sns.despine()


# In[133]:


fig, axs = plt.subplots(figsize=(40,5))
sns.countplot(x='country', hue='contract_type', data=df).set_title("Raw Column",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
sns.despine()


# In[134]:


fig, axs = plt.subplots(figsize=(40,5))
sns.countplot(x='value', hue='contract_type', data=df).set_title("Raw Column",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
sns.despine()


# In[135]:


# Client as JV?
change = {'No':0,'Yes':1}
df.jv = df.jv.map(change)

# Business Unit
change = {'FPS':0,'OPE':1,'PRE':2,'PTE':3,'PTW':4,'RNZ':5}
df.business_unit = df.business_unit.map(change)

# contract type
change = {'BOT/BOOT/BOM':0,'Decommissioning':1,'Engineering':2,'Engineering & Construction':3,'Operations & Maintenance':4,'Others':5,'Specialist Technical Services':6,'Training Services':7}
df.contract_type = df.contract_type.map(change)


corr_df = df.corr()
fig, axs = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df).set_title("Correlation Map",fontdict= { 'fontsize': 20, 'fontweight':'bold'});


# In[67]:



df = pd.read_excel("pv.xlsx")
df = df.replace(np.nan, '', regex=True)
df = df.fillna('')
df.set_index("contract_name")
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.01, random_state=25)

print(f"No. of training examples: {train.shape[0]}")
print(f"No. of testing examples: {test.shape[0]}")

df["contract_name"] = df.index
df = pd.get_dummies(df, columns=['jv','business_unit'])
df.drop(['sn','account_name','award_date','country','value', 'duration'], axis = 1, inplace = True)
df.columns


# In[68]:


df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


# In[69]:


df.columns


# In[70]:


# I splitted df to train and test
train, test = df.loc[train.index], df.loc[test.index]

X_train = train.drop(["contract_name",'contract_type'], axis = 1)
Y_train = train["contract_type"]
train_names = X_train.columns

X_test = test.drop(["contract_name",'contract_type'], axis = 1)


# In[71]:


corr_train = X_train.corr()
fig, axs = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_train).set_title("Correlation Map",fontdict= { 'fontsize': 20, 'fontweight':'bold'});
plt.show()


# In[72]:


X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_predDT = decision_tree.predict(X_test)

print("Accuracy of the model: ",round(decision_tree.score(X_train, Y_train) * 100, 2))


# In[34]:


importances = pd.DataFrame(decision_tree.feature_importances_, index = train_names)
importances.sort_values(by = 0, inplace=True, ascending = False)
importances = importances.iloc[0:6,:] 

plt.figure(figsize=(8, 5)) 
sns.barplot(x=0, y=importances.index, data=importances,palette="deep").set_title("Feature Importances",
                                                                                 fontdict= { 'fontsize': 20,
                                                                                            'fontweight':'bold'});
sns.despine()


# In[36]:


submit = pd.DataFrame({"contract_name":test.contract_name, 'contract_type':Y_predDT.ravel()})
submit.to_csv("result.csv",index = False)


# In[ ]:





# In[ ]:




