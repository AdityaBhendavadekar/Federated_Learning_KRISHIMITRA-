import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_path = '/media/aditya/Work/B-Tech/EDI Flask/fertilizer_prediction_model.pkl'  # Adjust the file path if needed
model = joblib.load(model_path)

print("Model loaded successfully!")

# Load the dataset again for testing
test_data_path = '/media/aditya/Work/B-Tech/EDI Flask/fertilizer_prediction_model.pkl'  # Adjust the file path if needed
df_test = pd.read_csv(test_data_path)

# Ensure the column names match your dataset
categorical_features = ['Soil Type', 'Crop Type']  # Update these based on your dataset
numerical_features = ['Nitrogen', 'Potassium', 'Phosphorous']  # Updated names for N, P, K
target_column = 'Fertilizer Name'

# Encode the target variable (if applicable, for comparison with predictions)
label_encoder = LabelEncoder()
df_test[target_column] = label_encoder.fit_transform(df_test[target_column])

# Prepare the test features and target
X_test = df_test[numerical_features + categorical_features]
y_test = df_test[target_column]

# Convert categorical features using pandas get_dummies
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)

# Align test data to the trained model's feature set
X_test_encoded = X_test_encoded.reindex(columns=model.get_booster().feature_names, fill_value=0)

# Make predictions
y_pred = model.predict(X_test_encoded)

# Decode the predictions to original class labels
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# Evaluate the model on the test set
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Test with a sample input
sample_input = pd.DataFrame({
    'Nitrogen': [50],
    'Potassium': [20],
    'Phosphorous': [30],
    'Soil Type': ['Clay'],  # Ensure this value exists in your training data
    'Crop Type': ['Rice']   # Ensure this value exists in your training data
})

# Encode the sample input
sample_input_encoded = pd.get_dummies(sample_input, columns=categorical_features, drop_first=True)
sample_input_encoded = sample_input_encoded.reindex(columns=model.get_booster().feature_names, fill_value=0)

# Predict fertilizer recommendation for the sample input
sample_prediction = model.predict(sample_input_encoded)
sample_prediction_decoded = label_encoder.inverse_transform(sample_prediction)

print("\nSample Input Prediction:")
print(f"Recommended Fertilizer: {sample_prediction_decoded[0]}")
