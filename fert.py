import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('fertilizer_prediction_model.pkl')
print("Model loaded successfully!")
    
# Load the LabelEncoder used during training
label_encoder = LabelEncoder()
label_encoder.classes_ = ['FertilizerA', 'FertilizerB', 'FertilizerC']  # Replace with your classes from training

# Define the prediction function
def predict_fertilizer(soil_type, crop_type, nitrogen, potassium, phosphorous):
    # Create a single input dataframe
    input_data = pd.DataFrame({
        'Soil Type': [soil_type],
        'Crop Type': [crop_type],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous]
    })

    # Categorical features used during training
    categorical_features = ['Soil Type', 'Crop Type']
    # Numerical features
    numerical_features = ['Nitrogen', 'Potassium', 'Phosphorous']

    # One-hot encode the input data
    input_encoded = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)

    # Align input data with model training columns
    input_encoded = input_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_encoded)
    predicted_label = label_encoder.inverse_transform([prediction[0]])[0]  # Decode the numeric label
    return predicted_label

# Take user input
soil_type = input("Enter Soil Type: ")
crop_type = input("Enter Crop Type: ")
nitrogen = float(input("Enter Nitrogen value: "))
potassium = float(input("Enter Potassium value: "))
phosphorous = float(input("Enter Phosphorous value: "))

# Predict and display the result
result = predict_fertilizer(soil_type, crop_type, nitrogen, potassium, phosphorous)
print(f"Recommended Fertilizer: {result}")
