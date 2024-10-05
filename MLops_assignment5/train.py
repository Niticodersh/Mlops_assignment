import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import requests
import os

# Set the tracking URI (replace 'your_tracking_uri' with your actual tracking URI)
mlflow.set_tracking_uri("https://dagshub.com/Niticodersh/Mlops_assignment.mlflow")  # Example for a local MLflow server

# Create a directory for saving plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Step 1: Download the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
response = requests.get(url)

with open("BostonHousing.csv", "wb") as file:
    file.write(response.content)

print("Dataset downloaded successfully!")

# Step 2: Load the dataset
data = pd.read_csv("BostonHousing.csv")

# Step 3: Display first few rows and summary statistics
print("First 5 rows of dataset:")
print(data.head())

print("Summary statistics of dataset:")
print(data.describe())

print("Missing values check:")
print(data.isnull().sum())

# Step 4: Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
correlation_heatmap_path = 'plots/correlation_heatmap.png'
plt.savefig(correlation_heatmap_path)  # Save the heatmap image
plt.close()

# Step 5: Prepare data for training
X = data.drop(columns=['medv'])  # Features
y = data['medv']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLflow
mlflow.start_run()

# Step 6: Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # Train the model
y_pred_lr = lr_model.predict(X_test)  # Make predictions

# Evaluate Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Create a directory for Linear Regression model artifacts
if not os.path.exists('artifacts/linear_regression'):
    os.makedirs('artifacts/linear_regression')

# Log parameters and metrics for Linear Regression model
mlflow.log_param("model_type", "Linear Regression")
mlflow.log_metric("mse", mse_lr)

# Define signature for model logging
signature = mlflow.models.signature.infer_signature(X_train, y_train)

# Get the tracking URI type
tracking_url_type_store = mlflow.get_tracking_uri()

# Log the Linear Regression model conditionally based on tracking URL type
if tracking_url_type_store != "file":
    mlflow.sklearn.log_model(
        lr_model, "model", registered_model_name="LinearRegressionBostonHousing", signature=signature
    )
else:
    mlflow.sklearn.log_model(lr_model, "model", signature=signature)

# Save Linear Regression predictions plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test['rm'], y_test, color='blue', label='Actual Prices')
plt.scatter(X_test['rm'], y_pred_lr, color='red', label='Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted House Prices')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price')
plt.legend()
linear_regression_plot_path = 'artifacts/linear_regression/actual_vs_predicted.png'
plt.savefig(linear_regression_plot_path)
plt.close()

# Log the Linear Regression plot
mlflow.log_artifact(linear_regression_plot_path)

# End the first run
mlflow.end_run()

# Start a new run for Random Forest
mlflow.start_run()

# Step 7: Train Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)  # Train the model
y_pred_rf = rf_model.predict(X_test)  # Make predictions

# Evaluate Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Create a directory for Random Forest model artifacts
if not os.path.exists('artifacts/random_forest'):
    os.makedirs('artifacts/random_forest')

# Log parameters and metrics for Random Forest model
mlflow.log_param("model_type", "Random Forest")
mlflow.log_metric("mse", mse_rf)

# Log the Random Forest model conditionally based on tracking URL type
if tracking_url_type_store != "file":
    mlflow.sklearn.log_model(
        rf_model, "model", registered_model_name="RandomForestBostonHousing", signature=signature
    )
else:
    mlflow.sklearn.log_model(rf_model, "model", signature=signature)

# Save Random Forest predictions plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test['rm'], y_test, color='blue', label='Actual Prices')
plt.scatter(X_test['rm'], y_pred_rf, color='red', label='Predicted Prices')
plt.title('Random Forest: Actual vs Predicted House Prices')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price')
plt.legend()
random_forest_plot_path = 'artifacts/random_forest/actual_vs_predicted.png'
plt.savefig(random_forest_plot_path)
plt.close()

# Log the Random Forest plot
mlflow.log_artifact(random_forest_plot_path)

# Log the correlation heatmap plot again for both models
mlflow.log_artifact(correlation_heatmap_path)

# End the second MLflow run
mlflow.end_run()

# Step 10: Compare models and save the best one
best_model = None
best_model_name = ""
if mse_lr < mse_rf:
    best_model = lr_model
    best_model_name = "linear_regression_model"
    print("\nBest model: Linear Regression")
else:
    best_model = rf_model
    best_model_name = "random_forest_model"
    print("\nBest model: Random Forest")

# Step 11: Save the best model
mlflow.start_run()  # Start a new run for saving the best model
mlflow.sklearn.log_model(best_model, "best_model")

# Step 12: Register the best model in the MLflow Model Registry
model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model"
mlflow.register_model(model_uri, best_model_name)

# Step 13: End the run
mlflow.end_run()

# Step 14: Print comparison results
print("\nComparison of Models:")
print(f"Linear Regression MSE: {mse_lr}")
print(f"Random Forest MSE: {mse_rf}")
