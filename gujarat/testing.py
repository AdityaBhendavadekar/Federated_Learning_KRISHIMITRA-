import pickle
import numpy as np

# Load the saved global model
with open("global_model.pkl", "rb") as model_file:
    global_model = pickle.load(model_file)

# Load the saved label encoder
with open("/media/aditya/Work/B-Tech/EDI Flask/server/label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define example input values
# Replace these values with your test case
# ,85,58,41,21.77046169,80.31964408,7.038096361,226.6555374

# 53,67,17,31.77681682,69.01852894,7.296972161,61.46892873,Apple

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