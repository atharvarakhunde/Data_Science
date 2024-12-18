# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
print("Loading the dataset...")
file_path = 'IMDb Movies India.csv'
data = pd.read_csv(file_path, encoding='latin1')
print("Dataset loaded successfully!\n")

# Step 2: Data Cleaning
print("Cleaning the dataset...")

# Clean column names
data.columns = data.columns.str.strip().str.replace(" ", "_")

# Convert Year to numeric
data['Year'] = data['Year'].str.extract(r'(\d{4})').astype(float)

# Convert Duration to numeric
data['Duration'] = data['Duration'].str.replace('min', '').str.strip().astype(float)

# Clean Votes column
data['Votes'] = data['Votes'].str.replace(',', '').str.extract(r'(\d+)').astype(float)

# Drop rows with missing target values
data = data.dropna(subset=['Rating'])

# Fill missing values for other columns
data['Genre'] = data['Genre'].fillna('Unknown')
data['Director'] = data['Director'].fillna('Unknown')
data['Actor_1'] = data['Actor_1'].fillna('Unknown')
data['Actor_2'] = data['Actor_2'].fillna('Unknown')
data['Actor_3'] = data['Actor_3'].fillna('Unknown')
data['Duration'] = data['Duration'].fillna(data['Duration'].median())
data['Year'] = data['Year'].fillna(data['Year'].median())

# Combine actor columns
data['Actors'] = data['Actor_1'] + ', ' + data['Actor_2'] + ', ' + data['Actor_3']
print("Dataset cleaned successfully!\n")

# Step 3: Feature Selection
print("Preparing features and target variable...")
features = data[['Genre', 'Director', 'Actors', 'Duration', 'Year', 'Votes']]
target = data['Rating']
print("Features and target variable prepared!\n")

# Step 4: One-Hot Encoding
print("Encoding categorical features...")
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = pd.DataFrame(
    encoder.fit_transform(features[['Genre', 'Director', 'Actors']]),
    index=features.index
)
encoded_features.columns = encoder.get_feature_names_out(['Genre', 'Director', 'Actors'])

# Combine numeric and encoded features
numeric_features = features[['Duration', 'Year', 'Votes']].fillna(0)
final_features = pd.concat([numeric_features, encoded_features], axis=1)
print("Categorical features encoded successfully!\n")

# Step 5: Split the Data
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(final_features, target, test_size=0.2, random_state=42)
print("Data split completed!\n")

# Step 6: Train the Model
print("Training the Random Forest model...")
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
print("Model trained successfully!\n")

# Step 7: Evaluate the Model
print("Evaluating the model...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model evaluation completed! Root Mean Squared Error (RMSE): {rmse:.2f}\n")

# Step 8: Predict on New Data
print("Predicting the rating for a new movie...")

# Example: Predicting for a new movie
new_movie_data = {
    'Genre': ['Comedy, Drama'],
    'Director': ['John Doe'],
    'Actors': ['Actor A, Actor B, Actor C'],
    'Duration': [120],
    'Year': [2023],
    'Votes': [5000]
}
new_df = pd.DataFrame(new_movie_data)

# Encode new movie data
encoded_new = pd.DataFrame(
    encoder.transform(new_df[['Genre', 'Director', 'Actors']]),
    index=new_df.index,
    columns=encoder.get_feature_names_out(['Genre', 'Director', 'Actors'])
)

# Combine numeric and encoded features
numeric_new = new_df[['Duration', 'Year', 'Votes']]
final_new_features = pd.concat([numeric_new, encoded_new], axis=1)

# Make prediction
predicted_rating = model.predict(final_new_features)
print(f"Predicted Rating for the new movie: {predicted_rating[0]:.2f}\n")
