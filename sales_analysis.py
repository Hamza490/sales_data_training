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
#print(model_1.summary())

# Model 2- Price
x2_train = sm.add_constant(x_train['Price'])  # Add a constant term to the model (intercept)
x2_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_2 = sm.OLS(y_train, x2_train).fit()
# Print the summary of the model
#print(model_2.summary())

# Model 3 - Price, Print_ads
x3_train = sm.add_constant(x_train[['Price', 'Print_ads']])  # Add a constant term to the model (intercept)
x3_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_3 = sm.OLS(y_train, x3_train).fit()
# Print the summary of the model
#print(model_3.summary())

# Model 4 - Price, Print_ads, online_ads
x4_train = sm.add_constant(x_train[['Price', 'Print_ads', 'online_ads']])  # Add a constant term to the model (intercept)
x4_test = sm.add_constant(x_test)
# Fit the model using OLS (Ordinary Least Squares)
model_4 = sm.OLS(y_train, x4_train).fit()
# Print the summary of the model
#print(model_4.summary())

# Step 6: Adding interaction terms for pairwise interaction models
def add_interaction_terms(x):
  interactions = {}
  for i in range(len(x.columns)):
        for j in range(i + 1, len(x.columns)):
            feature1 = x.columns[i]
            feature2 = x.columns[j]
            interaction_name = f"{feature1}_x_{feature2}"
            interactions[interaction_name] = x[feature1] * x[feature2]
  '''
  print("Interaction Terms Names:")
  for name in interactions.keys():
      print(name)'''
  return pd.DataFrame(interactions)

def fit_and_plot_model(X_train, y_train, X_test, y_test, model_number):
    X_train_const = sm.add_constant(X_train)  # Add a constant term to the model (intercept)
    model = sm.OLS(y_train, X_train_const).fit()
    
    # Predictions and R^2 calculation
    y_pred = model.predict(sm.add_constant(X_test))
    r2 = r2_score(y_test, y_pred)
    
    # Plotting
    plt.figure()
    plt.title(f'Model {model_number}: R² = {r2:.4f}')
    plt.scatter(y_test, y_pred, label='Predicted vs Actual')
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='r')
    plt.legend()
    plt.show()

    return model, r2

# Adding interaction terms and fitting for Model 1 (All Variables)
x1_interaction_train = add_interaction_terms(x1_train)
x1_interaction_test = add_interaction_terms(x1_test)
model_1_interaction, r2_model_1 = fit_and_plot_model(x1_interaction_train, y_train, x1_interaction_test, y_test, model_number=1)

# Adding interaction terms and fitting for Model 2 (Price)
#x2_interaction_train = add_interaction_terms(x2_train)
#x2_interaction_test = add_interaction_terms(x2_test)
#model_2_interaction, r2_model_2 = fit_and_plot_model(x2_interaction_train, y_train, x2_interaction_test, y_test, model_number=2)

# Adding interaction terms and fitting for Model 3 (Price, Print_ads)
x3_interaction_train = add_interaction_terms(x3_train)
x3_interaction_test = add_interaction_terms(x3_test)
model_3_interaction, r2_model_3 = fit_and_plot_model(x3_interaction_train, y_train, x3_interaction_test, y_test, model_number=3)

# Adding interaction terms and fitting for Model 4 (Price, Print_ads, online_ads)
x4_interaction_train = add_interaction_terms(x4_train)
x4_interaction_test = add_interaction_terms(x4_test)
model_4_interaction, r2_model_4 = fit_and_plot_model(x4_interaction_train, y_train, x4_interaction_test, y_test, model_number=4)
