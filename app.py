from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

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
    

    

    

if __name__ == '__main__':
    app.run(debug=True)
