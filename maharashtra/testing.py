import pickle
import numpy as np

# Load the saved global model
with open("maharashtra/updated_client_model_mah.pkl", "rb") as model_file:
    global_model = pickle.load(model_file)

# Load the saved label encoder
with open("maharashtra/label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)


example_input = {
    'N_SOIL': 53,
    'P_SOIL': 67,
    'K_SOIL': 17,
    'TEMPERATURE': 31.77,
    'HUMIDITY': 69.01,
    'ph': 7.2,
    'RAINFALL': 61.4
}

# Prepare the input as a numpy array
example_features = np.array([[example_input['N_SOIL'],
                               example_input['P_SOIL'],
                               example_input['K_SOIL'],
                               example_input['TEMPERATURE'],
                               example_input['HUMIDITY'],
                               example_input['ph'],
                               example_input['RAINFALL']]])

# Make a prediction
predicted_label = global_model.predict(example_features)[0]

# Convert the predicted label back to the crop name
predicted_crop = label_encoder.inverse_transform([predicted_label])[0]

print(f"Predicted Crop: {predicted_crop}")