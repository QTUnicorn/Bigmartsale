import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load train and test data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Print first 5 rows of training data
print('First 5 rows of training data:\n', train_data.head())

# Print shape of training and test data
print('\nShape of training data:', train_data.shape)
print('Shape of testing data:', test_data.shape)

# Split training data into features (train_x) and target (train_y)
train_x = train_data.drop(columns=['Item_Outlet_Sales'], axis=1)
train_y = train_data['Item_Outlet_Sales']

# Split test data into features (test_x) and target (test_y)
test_x = test_data.drop(columns=['Item_Outlet_Sales'], axis=1)
test_y = test_data['Item_Outlet_Sales']

# Train linear regression model on training data
model = LinearRegression()
model.fit(train_x, train_y)

# Print coefficients and intercept of trained model
print('\nCoefficients of model:', model.coef_)
print('Intercept of model:', model.intercept_)

# Make predictions on training data and calculate RMSE
predict_train = model.predict(train_x)
rmse_train = mean_squared_error(train_y, predict_train)**(0.5)
print('\nPredicted Item_Outlet_Sales on training data:\n', predict_train)
print('RMSE on training dataset:', rmse_train)

# Make predictions on test data and calculate RMSE
predict_test = model.predict(test_x)
rmse_test = mean_squared_error(test_y, predict_test)**(0.5)
print('\nPredicted Item_Outlet_Sales on test data:\n', predict_test)
print('RMSE on test dataset:', rmse_test)