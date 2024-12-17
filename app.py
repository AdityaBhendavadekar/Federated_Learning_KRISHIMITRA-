from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Paths for Maharashtra and Gujarat
STATE_PATHS = {
    'maharashtra': {
        'model': 'maharashtra/global_model.pkl',
        'label_encoder': 'maharashtra/label_encoder.pkl'
    },
    'gujarat': {
        'model': 'gujarat/global_model.pkl',
        'label_encoder': 'gujarat/label_encoder.pkl'
    }
}

def load_model_and_encoder(state):
    """Load the model and label encoder for a given state."""
    try:
        model_path = STATE_PATHS[state]['model']
        label_encoder_path = STATE_PATHS[state]['label_encoder']

        # Load the saved model
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Load the saved label encoder
        with open(label_encoder_path, 'rb') as encoder_file:
            label_encoder = pickle.load(encoder_file)

        return model, label_encoder
    except Exception as e:
        raise Exception(f"Error loading model or label encoder: {e}")

@app.route('/', methods=['GET'])
def hello():
    return render_template("index.html")


@app.route('/api/predict/<state>', methods=['POST'])
def predict(state):
    """Predict the top 5 crops for a specific state."""
    if state not in STATE_PATHS:
        return jsonify({'error': f"Invalid state '{state}'. Valid states are: {list(STATE_PATHS.keys())}"}), 400

    try:
        # Load the model and label encoder
        model, label_encoder = load_model_and_encoder(state)

        # Parse the input data
        input_data = request.json
        features = [
            input_data['N_SOIL'], input_data['P_SOIL'], input_data['K_SOIL'],
            input_data['TEMPERATURE'], input_data['HUMIDITY'], input_data['ph'],
            input_data['RAINFALL']
        ]

        # Prepare the input as a numpy array
        example_features = np.array([features])

        # Make a prediction: Get probabilities for each crop
        probabilities = model.predict_proba(example_features)[0]

        # Get the top 5 crops with the highest probabilities
        top_indices = np.argsort(probabilities)[-5:][::-1]  # Indices of the top 5 crops
        top_crops = label_encoder.inverse_transform(top_indices)  # Get crop names
        top_probs = probabilities[top_indices]  # Corresponding probabilities

        # Prepare the response
        result = [{'crop': crop, 'probability': round(prob, 4)} for crop, prob in zip(top_crops, top_probs)]

        return jsonify({'predicted_crops': result})
    except KeyError as e:
        return jsonify({'error': f"Missing input parameter: {str(e)}"}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/add-entry', methods=['POST'])
def add_entry():
    try:
        data = request.json

        # Validate the incoming data
        required_fields = ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL', 'CROP']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Load existing dataset
        df = pd.read_csv("maharashtra/maharashtra_data.csv")

        # Add the new entry
        new_entry = {
            'N_SOIL': data['N_SOIL'],
            'P_SOIL': data['P_SOIL'],
            'K_SOIL': data['K_SOIL'],
            'TEMPERATURE': data['TEMPERATURE'],
            'HUMIDITY': data['HUMIDITY'],
            'ph': data['ph'],
            'RAINFALL': data['RAINFALL'],
            'CROP': data['CROP']
        }
        # df = df.append(new_entry, ignore_index=True)

        # Convert the new entry into a DataFrame
        new_entry_df = pd.DataFrame([new_entry])
        # Use concat for appending
        df = pd.concat([df, new_entry_df], ignore_index=True)

        # Save the updated dataset back to the file
        df.to_csv("maharashtra/maharashtra_data.csv", index=False)

        return jsonify({"message": "Entry added successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/fert', methods=['POST'])
def predict_fertilizer():
    # Load the saved model and tools
    model = joblib.load("recommendation_model/fertilizer_model.pkl")
    scaler = joblib.load("recommendation_model/scaler.pkl")
    label_encoders = joblib.load("recommendation_model/label_encoders.pkl")
    target_encoder = joblib.load("recommendation_model/target_encoder.pkl")
    """
    Predict the fertilizer based on custom input data provided in JSON format.
    """
    try:
        # Parse input data from JSON request
        custom_data = request.json

        # Validate input data
        required_keys = [
            "Temparature", "Humidity", "Moisture", 
            "Soil Type", "Crop Type", "Nitrogen", 
            "Potassium", "Phosphorous"
        ]
        if not all(key in custom_data for key in required_keys):
            return jsonify({"error": "Missing required fields in input"}), 400

        # Extract and preprocess numerical data
        numerical_features = ["Temparature", "Humidity", "Moisture", "Nitrogen", "Potassium", "Phosphorous"]
        numerical_data = np.array([custom_data[feature] for feature in numerical_features]).reshape(1, -1)
        numerical_data_scaled = scaler.transform(numerical_data)

        # Extract and preprocess categorical data
        categorical_features = ["Soil Type", "Crop Type"]
        categorical_data = []
        for feature in categorical_features:
            if custom_data[feature] in label_encoders[feature].classes_:
                encoded_value = label_encoders[feature].transform([custom_data[feature]])[0]
            else:
                return jsonify({
                    "error": f"Unseen value '{custom_data[feature]}' for feature '{feature}'. Valid values are: {list(label_encoders[feature].classes_)}"
                }), 400
            categorical_data.append(encoded_value)
        categorical_data = np.array(categorical_data).reshape(1, -1)

        # Combine numerical and categorical data
        processed_data = np.hstack((numerical_data_scaled, categorical_data))

        # Predict fertilizer
        prediction = model.predict(processed_data)
        predicted_fertilizer = target_encoder.inverse_transform(prediction)[0]

        # Return prediction
        return jsonify({"predicted_fertilizer": predicted_fertilizer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
