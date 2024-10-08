# Step 1 - Import Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Step 2: Load Dataset into Pandas DataFrame
dataFrame = pd.read_csv('Sales_Data.csv')

print(dataFrame.info()) #Checks to see if we have any blank cells in our dataset
print(dataFrame.describe()) #Churns out high level stas from your dataset