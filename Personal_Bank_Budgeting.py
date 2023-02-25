#!/usr/bin/env python
# coding: utf-8

# In[82]:


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
get_ipython().system('pip install pandas ')

get_ipython().system('pip install scikit-learn')

get_ipython().system('pip install streamlit')


st.title('Uber pickups in NYC')


# In[81]:


# Wells Fargo Given DataSet

# In[4]:


df = pd.read_csv(
    '/Users/eric/Documents/Github/Personal_Bank_Budgeting/Checking1.csv')
df.head()

# import my data set


# General Bank Statement
#
# - Includes money earned and spent

# In[5]:


df.rename(columns={'12/30/22': 'Dates'}, inplace=True)
df.rename(columns={'-17': 'Money Spent'}, inplace=True)
df.rename(columns={'Venmo': 'Type of Transaction'}, inplace=True)
df.rename(columns={
          'PURCHASE AUTHORIZED ON 12/30 VENMO* Visa Direct NY S302364781105329 CARD 7605': 'Transaction'}, inplace=True)
del df['*']
df.head()

# rename columns to make more sense, remove null column '*'


# In[6]:


df


# In[7]:


df['Money Spent'] = df['Money Spent'].multiply(-1)


df.head()

# multiply all values by negative 1, to show the value spent as a positive value

df


# In[8]:


df.columns


# In[9]:


df.describe()

# stats of money spent


# In[ ]:


# In[10]:


df['Dates'] = pd.to_datetime(df['Dates'])


df['Month'] = df['Dates'].dt.month


df


# putting the dates into date time format, and extracting the month to have a month by month analysis


# In[11]:


month = df.groupby('Month')

month.mean()

month.describe()

# stats per month


# In[12]:


plt.figure(figsize=(12, 8))
plt.scatter(data=df, x='Month', y='Money Spent')

# scatter plot of Money Spent vs Month


# In[13]:


x = df['Month']
y = df['Money Spent']
x.corr(y)

# getting a correlation coefficent between x and y, clearly no linear relationship between Month and money spent


# In[14]:


plt.figure(figsize=(15, 8))
sns.histplot(x='Money Spent', data=df, bins=20, kde=True, stat='density')

# histogram of money spent


# In[15]:


fig = px.line(df, x='Dates', y="Money Spent")
fig.show()

# time series plot of money spent vs dates


# In[16]:


fig = px.line(df, x='Month', y="Money Spent")
fig.show()


# In[17]:


X = df['Month'].values.reshape(-1, 1)
Y = df['Money Spent'].values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

# linear regression model, almost constant


N = len(X)


x_mean = X.mean()
y_mean = Y.mean()

B1_num = ((x - x_mean) * (y - y_mean)).sum()
B1_den = ((x - x_mean)**2).sum()
B1 = B1_num / B1_den

B0 = y_mean - (B1*x_mean)


print("Regression Line Equation=", B1, "X +", B0)


# In[18]:


df.groupby('Type of Transaction')['Money Spent'].mean()


# In[19]:


df.groupby('Type of Transaction')['Money Spent'].sum()


# In[20]:


df['Type of Transaction'].value_counts()


# In[21]:


plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.countplot(x='Type of Transaction', data=df)


# In[22]:


fig = px.histogram(df, x='Type of Transaction',
                   title="Distribution of the Types of Transactions")

fig.update_xaxes(categoryorder='total ascending')


fig.show()  # to actually show the histogram


# In[23]:


plt.figure(figsize=(15, 8))
sns.boxplot(data=df, x="Month", y='Money Spent')

# compare the distributions of money spent for each month


# In[24]:


df_new = df.loc[:, ['Month', 'Money Spent']]
plt.figure(figsize=(15, 8))
sns.pairplot(df_new)


# In[25]:


df.groupby('Month')['Money Spent'].sum()


# Spending Bank Statement
#
# - Includes only my money spent

# In[26]:


df_spend = pd.read_csv(
    '/Users/eric/Documents/Github/Personal_Bank_Budgeting/Checking1.csv')
df_spend.head()


# In[27]:


df_spend.rename(columns={'12/30/22': 'Dates'}, inplace=True)
df_spend.rename(columns={'-17': 'Money Spent'}, inplace=True)
df_spend.rename(columns={'Venmo': 'Type of Transaction'}, inplace=True)
df_spend.rename(columns={
                'PURCHASE AUTHORIZED ON 12/30 VENMO* Visa Direct NY S302364781105329 CARD 7605': 'Transaction'}, inplace=True)
del df_spend['*']


df_spend['Money Spent'] = df_spend['Money Spent'].multiply(-1)


df_spend.head()
df_spend

len(df_spend)

# same code as before, rename all column names and
# multiply by -1 to represent spent money as positive


# In[28]:


for x in df_spend.index:
    if (df.loc[x, 'Money Spent'] < 0):
        df_spend = df_spend.drop(x)


len(df_spend)

# loop through each row, if money spent is negative(aka money earned)
# delete the value from the dataset


df_spend


# In[29]:


df_spend.describe()


# In[30]:


df_spend['Dates'] = pd.to_datetime(df_spend['Dates'])


df_spend['Month'] = df_spend['Dates'].dt.month

df_spend


# In[31]:


plt.figure(figsize=(12, 8))
plt.scatter(data=df_spend, x='Month', y='Money Spent')


# In[32]:


plt.figure(figsize=(15, 8))
sns.histplot(x='Money Spent', data=df_spend, bins=20, kde=True, stat='density')


# In[33]:


# remove outliers

for x in df_spend.index:
    if df_spend.loc[x, 'Money Spent'] > 200:
        df_spend = df_spend.drop(x)
len(df_spend)


# In[34]:


plt.figure(figsize=(12, 8))
plt.scatter(data=df_spend, x='Month', y='Money Spent')


# In[35]:


plt.figure(figsize=(15, 8))
sns.histplot(x='Money Spent', data=df_spend, bins=20, kde=True, stat='density')


# In[36]:


fig = px.line(df_spend, x='Dates', y="Money Spent")
fig.show()


# In[37]:


fig = px.line(df_spend, x='Month', y="Money Spent")
fig.show()


# In[38]:


df_new2 = df_spend.loc[:, ['Month', 'Money Spent']]
plt.figure(figsize=(15, 8))
sns.pairplot(df_new2)


# In[39]:


sns.barplot(data=df_spend, x="Month", y="Money Spent")


# In[40]:


X = df_spend['Month'].values.reshape(-1, 1)
Y = df_spend['Money Spent'].values.reshape(-1, 1)

linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

# linear regression model, almost constant


N = len(X)


x_mean = X.mean()
y_mean = Y.mean()

B1_num = ((x - x_mean) * (y - y_mean)).sum()
B1_den = ((x - x_mean)**2).sum()
B1 = B1_num / B1_den

B0 = y_mean - (B1*x_mean)


print("Regression Line Equation=", B1, "X +", B0)


# Mint.com Given Data Set
#
# -Better Data that is easier to read
# -More categories already labeled

# In[41]:


df_mint = pd.read_csv(
    '/Users/eric/Documents/Github/Personal_Bank_Budgeting/transactions.csv')

# upload new dataset from mint


# In[42]:


df_mint.head()


# In[43]:


df_mint.describe()


# In[44]:


for x in df_mint.index:
    if df_mint.loc[x, 'Category'] == 'Income' or df_mint.loc[x, 'Category'] == 'Paycheck':
        df_mint = df_mint.drop(x)


len(df_mint)

# drop every time I make income, only concerned with spending


# In[68]:


df_mint = df_mint.loc[::-1]

fig = px.line(df_mint, x='Date', y="Amount", title="Money Spent Through Time")
fig.show()


# In[ ]:


df_mint['Date'] = pd.to_datetime(df_mint['Date'])


df_mint['Month'] = df_mint['Date'].dt.month

df_mint


# In[ ]:


df_mint.groupby(['Category', 'Amount']).mean()


# In[69]:


df_mint.groupby('Category')['Amount'].sum().sort_values(ascending=False)


# In[ ]:


df_mint['Category'].value_counts()


# In[ ]:


plt.figure(figsize=(15, 8))
plt.xticks(rotation=90)
sns.countplot(x='Category', data=df_mint)


# In[ ]:


fig = px.histogram(df_mint, x='Category',
                   title="Distribution of the Types of Transactions")

fig.update_xaxes(categoryorder='total ascending')


fig.show()  # to actually show the histogram


# In[ ]:


# Overall takeways from the project
#
# - There is no clear linear relationship between month and money spent
# - I do most of my purchases at transfers (Zelle or Venmo), fast food, and restaurants
# - Most of the money value is spent on transfers, rent, and shopping
# - Most of my spending is done around summer and winter (During holiday breaks from school when I go back home)
# - The amount I have spent has increased recently (Rent and Utilities and Groceries is the main reason)
# - Overall looking at the linear regression model, my spending is consistent throughout the year with few big outliers that come (mostly from rent)
#
# What's next?
#
# - Try to learn how to work with APIS and real time data
# - Try to learn data cloud services
# - Learn machine learning principles to be able to apply models to my future projects
# - Learn SQL and retouch on my python
