import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the global dataset
global_csv_path = "india_data.csv"  # Replace with your server's dataset path
global_data = pd.read_csv(global_csv_path)

# Define features and target
features = ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']
target = 'CROP'

# Drop the 'STATE' column
global_data = global_data.drop(columns=['STATE'])

# Label encoding for the target (CROP)
label_encoder = LabelEncoder()
global_data[target] = label_encoder.fit_transform(global_data[target])

# Define the feature matrix (X) and target vector (y) for training the global model
X_global = global_data[features]
y_global = global_data[target]

# Train the global model using Logistic Regression
global_model = LogisticRegression(max_iter=1000)
global_model.fit(X_global, y_global)

# Define the folder path
# server_folder = "/"
# os.makedirs(server_folder, exist_ok=True)  # Ensure the 'server' folder exists

# Save the global model and label encoder in the 'server' folder
model_file_path = os.path.join( "global_model.pkl")
label_encoder_file_path = os.path.join( "label_encoder.pkl")

with open(model_file_path, "wb") as model_file:
    pickle.dump(global_model, model_file)

with open(label_encoder_file_path, "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print(f"Global model saved at: {model_file_path}")
print(f"Label encoder saved at: {label_encoder_file_path}")
