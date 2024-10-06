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
import os 

# Set the experiment name for MLflow
mlflow.set_experiment("BostonHousingExperiment")

# Create a directory for saving plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

print("First 5 rows of dataset:")
print(data.head())

print("Summary statistics of dataset:")
print(data.describe())

print("Missing values check:")
print(data.isnull().sum())

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
correlation_heatmap_path = 'plots/correlation_heatmap.png'
plt.savefig(correlation_heatmap_path)
plt.close()

# Prepare features and target variable
X = data.drop(columns=['medv'])  # Features
y = data['medv']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run for Linear Regression
with mlflow.start_run():
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)  # Train the model
    y_pred_lr = lr_model.predict(X_test)  # Make predictions

    # Evaluate the model
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f"Linear Regression MSE: {mse_lr}")

    # Log parameters and metrics
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_metric("mse", mse_lr)

    # Log the model
    mlflow.sklearn.log_model(lr_model, "linear_regression_model")

    # Save predictions plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test['rm'], y_test, color='blue', label='Actual Prices')
    plt.scatter(X_test['rm'], y_pred_lr, color='red', label='Predicted Prices')
    plt.title('Linear Regression: Actual vs Predicted House Prices')
    plt.xlabel('Average Number of Rooms (RM)')
    plt.ylabel('House Price')
    plt.legend()
    linear_regression_plot_path = 'plots/linear_regression_actual_vs_predicted.png'
    plt.savefig(linear_regression_plot_path)
    plt.close()

    # Log the plot
    mlflow.log_artifact(linear_regression_plot_path)

# Start MLflow run for Random Forest
with mlflow.start_run():
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)  # Train the model
    y_pred_rf = rf_model.predict(X_test)  # Make predictions

    # Evaluate the model
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f"Random Forest MSE: {mse_rf}")

    # Log parameters and metrics
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_metric("mse", mse_rf)

    # Log the model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    # Save predictions plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test['rm'], y_test, color='blue', label='Actual Prices')
    plt.scatter(X_test['rm'], y_pred_rf, color='red', label='Predicted Prices')
    plt.title('Random Forest: Actual vs Predicted House Prices')
    plt.xlabel('Average Number of Rooms (RM)')
    plt.ylabel('House Price')
    plt.legend()
    random_forest_plot_path = 'plots/random_forest_actual_vs_predicted.png'
    plt.savefig(random_forest_plot_path)
    plt.close()

    # Log the plot
    mlflow.log_artifact(random_forest_plot_path)

# Compare Models
print("\nComparison of Models:")
print(f"Linear Regression MSE: {mse_lr}")
print(f"Random Forest MSE: {mse_rf}")

# Determine the best model
best_model = lr_model if mse_lr < mse_rf else rf_model
best_model_name = "Linear Regression" if mse_lr < mse_rf else "Random Forest"
print(f"Best model: {best_model_name}")

# Save the best model
with mlflow.start_run():
    mlflow.sklearn.log_model(best_model, "best_model")
