# Step 1 - Import Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Step 2: Load Dataset into Pandas DataFrame
dataFrame = pd.read_csv('Sales_Data.csv')

# Step 3: Data Cleaning & High Level Stats
#print(dataFrame.info()) #Checks to see if we have any blank cells in our dataset
#print(dataFrame.describe()) #Churns out high level stas from your dataset

# Step 4: Splitting dataset into sales_test and sales_train
print(dataFrame.columns)
x = dataFrame[['Price', 'Print_ads', 'online_ads', 'TV_ads']]
y = dataFrame['Sales']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

'''
Tests for Step 4 (ensuring the data is split as expected)
print("Total samples:", len(x))
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))
'''

# Step 5: Build Initial Linear Regression Model (No Interactions)

# Model 1 - All Variables
x1_train = sm.add_constant(x_train)  # Add a constant term to the model (intercept)
x1_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_1 = sm.OLS(y_train, x1_train).fit()
# Print the summary of the model
print(model_1.summary())

# Model 2- Price
x2_train = sm.add_constant(x_train['Price'])  # Add a constant term to the model (intercept)
x2_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_2 = sm.OLS(y_train, x2_train).fit()
# Print the summary of the model
print(model_2.summary())

# Model 3 - Price, Print_ads
x3_train = sm.add_constant(x_train[['Price', 'Print_ads']])  # Add a constant term to the model (intercept)
x3_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_3 = sm.OLS(y_train, x3_train).fit()
# Print the summary of the model
print(model_3.summary())

# Model 4 - Price, Print_ads, online_ads
x4_train = sm.add_constant(x_train[['Price', 'Print_ads', 'online_ads']])  # Add a constant term to the model (intercept)
x4_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_4 = sm.OLS(y_train, x4_train).fit()
# Print the summary of the model
print(model_4.summary())

# Model 4 - Price, Print_ads, online_ads
x4_train = sm.add_constant(x_train[['Price', 'Print_ads', 'online_ads']])  # Add a constant term to the model (intercept)
x4_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_4 = sm.OLS(y_train, x4_train).fit()
# Print the summary of the model
print(model_4.summary())