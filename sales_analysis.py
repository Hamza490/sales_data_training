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
print("Total samples:", len(x))
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))