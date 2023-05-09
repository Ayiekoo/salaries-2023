#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing requried packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[2]:


df = "C:/Users/Ayieko/Desktop/python/walmart/Walmart_Data_Analysis_and_Forcasting.csv"
df = pd.read_csv(df)
print(df)


# In[4]:


df.head(10)


# In[6]:


# Create a boxplot of the weekly sales data
plt.boxplot(df['Weekly_Sales'])

# Add labels to the plot
plt.title('Weekly Sales Boxplot')
plt.ylabel('Sales (in millions)')
plt.xticks([1], ['Sales'])

# Display the plot
plt.show()


# In[11]:


# Group the data by store
sales_by_store = df.groupby('Store')

# Create a line chart of the weekly sales data for each store
for store, data in sales_by_store:
    plt.plot(data['Date'], data['Weekly_Sales'], label=f'Store {store}')

# Add labels to the plot
plt.title('Weekly Sales by Store')
plt.xlabel('Date')
plt.ylabel('Sales (in billions)')
plt.legend()

# Display the plot
plt.show()


# In[12]:


import seaborn as sns


# In[14]:


# Select the columns of interest
columns_of_interest = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
sales_data_subset = df[columns_of_interest]

# Compute the correlation matrix
corr_matrix = sales_data_subset.corr()

# Visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Add a title to the plot
plt.title('Correlation Matrix')

# Add legend


# Display the plot
plt.show()




# In[15]:


# Create a histogram of the weekly sales data for each store
df.hist(column='Weekly_Sales', by='Store', bins=20, figsize=(10,10))

# Add labels to the plot
plt.suptitle('Weekly Sales Distribution by Store')
plt.xlabel('Sales (in millions)')
plt.ylabel('Frequency')

# Display the plot
plt.show()


# In[17]:


# Create a scatter plot of weekly sales vs. temperature
plt.scatter(df['Temperature'], df['Weekly_Sales'])

# Add labels to the plot
plt.title('Weekly Sales vs. Temperature')
plt.xlabel('Temperature (in Fahrenheit)')
plt.ylabel('Sales (in millions)')

# Display the plot
plt.show()

# Create a scatter plot of weekly sales vs. fuel price
plt.scatter(df['Fuel_Price'], df['Weekly_Sales'])

# Add labels to the plot
plt.title('Weekly Sales vs. Fuel Price')
plt.xlabel('Fuel Price (in dollars per gallon)')
plt.ylabel('Sales (in millions)')

# Display the plot
plt.show()


# In[19]:


# Group the data by holiday flag and compute the mean weekly sales for each group
holiday_sales = df.groupby('Holiday_Flag')['Weekly_Sales'].mean()

# Create a bar chart of the mean weekly sales for each group
plt.bar(['Non-Holiday', 'Holiday'], holiday_sales)

# Add labels to the plot
plt.title('Mean Weekly Sales by Holiday')
plt.xlabel('Holiday Flag')
plt.ylabel('Mean Sales (in millions)')

# Display the plot
plt.show()


# In[20]:


# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort the data by date
df = df.sort_values(by='Date')

# Create a time-series plot of the weekly sales over time
plt.plot(df['Date'], df['Weekly_Sales'])

# Add labels to the plot
plt.title('Weekly Sales over Time')
plt.xlabel('Date')
plt.ylabel('Sales (in millions)')

# Display the plot
plt.show()


# In[ ]:




