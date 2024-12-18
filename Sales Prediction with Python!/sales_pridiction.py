import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

file_path = 'advertising.csv' 
data = pd.read_csv(file_path)

print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Information:")
print(data.info())


print("\nMissing values in each column:")
print(data.isnull().sum())

X = data.drop(columns=['Sales'])  # Features: TV, Radio, Newspaper
y = data['Sales']  # Target: Sales

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  #it shows the model is good at prediction or not


# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# For Making an image of easy understanding of actual and prediction sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)  # Line of perfect prediction
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid(True)
plt.savefig('actual_vs_predicted_sales.png')
print("\nThe plot has been saved as 'actual_vs_predicted_sales.png'")

# Save the trained model to a file for later use
joblib.dump(model, 'sales_prediction_model.pkl')
print("\nModel has been saved as 'sales_prediction_model.pkl'")

# Predict sales for new data
# Example for testing the model
new_data = pd.DataFrame({'TV': [200], 'Radio': [40], 'Newspaper': [10]})
new_sales_prediction = model.predict(new_data)
print(f"Predicted Sales: {new_sales_prediction[0]}")
