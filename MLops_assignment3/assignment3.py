import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import requests
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"

response = requests.get(url)

with open("BostonHousing.csv", "wb") as file:
        file.write(response.content)

print("Dataset downloaded successfully!")

data = pd.read_csv("BostonHousing.csv")

print("First 5 rows of dataset")
print(data.head())

print("Summary statistics of dataset")
print(data.describe())

print("Missing values check")
print(data.isnull().sum())

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
# Saving the heatmap image
plt.savefig('correlation_heatmap.png') 
plt.show()

sns.set(style="whitegrid") 
sns.scatterplot(x='rm', y='medv', data=data)

sns.regplot(x='rm', y='medv', data=data, scatter_kws={"s": 20}, line_kws={"color": "red"})
plt.title("Relationship between Average Rooms (RM) and Median Value (MEDV)")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median Value of Homes (MEDV)")

# Save the plot as a PNG file
plt.savefig("rm_vs_medv_plot.png")

plt.show()

X = data.drop(columns=['medv']) # Features
y = data['medv'] # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Model coefficients
print("Coefficient (slope):", model.coef_)
print("Intercept:", model.intercept_)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(X_test['rm'], y_test, color='blue', label='Actual Prices')
plt.scatter(X_test['rm'], y_pred, color='red', label='Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price')
plt.legend()

# Saving the actual vs predicted price plot as an image
plt.savefig('actual_vs_predicted_prices.png') 
plt.show()





