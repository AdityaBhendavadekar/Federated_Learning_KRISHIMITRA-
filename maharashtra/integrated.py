from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.oauth2 import service_account
import os
import io
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


# Service account credentials file path
SERVICE_ACCOUNT_FILE = '/media/aditya/Work/Drive/edi-models-86ad22fb54fe.json'
SCOPES = ['https://www.googleapis.com/auth/drive']

# Folder IDs
DOWNLOAD_FOLDER_ID = '1IemFVUnKCwG21zriWhbwYZ63QFreuoeL'  # Replace with your folder ID for downloading
UPLOAD_FOLDER_ID = '1ahPF7jzmGTTePpKSuNSDqsUabhQCwaZR'  # Replace with your folder ID for uploading

# File names
LOCAL_CSV_PATH = "/media/aditya/Work/Drive/client/maharashtra_data.csv"  # Path to the dataset
LABEL_ENCODER_FILE = "/media/aditya/Work/Drive/client/label_encoder.pkl"  # Path to the label encoder
GLOBAL_MODEL_FILE = 'global_model.pkl'
UPDATED_MODEL_FILE = 'updated_client_model_mah.pkl'
fine_tuned_global_model_path = "global_model1.pkl"  # Path to save the fine-tuned global model


# Authenticate and initialize the Drive API client
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

drive_service = build('drive', 'v3', credentials=credentials)

def download_file(file_name, folder_id, save_path):
    """Download a specific file from Google Drive."""
    try:
        query = f"'{folder_id}' in parents and name='{file_name}'"
        results = drive_service.files().list(q=query).execute()
        files = results.get('files', [])

        if not files:
            print(f"File '{file_name}' not found in the folder.")
            return False

        # Download the file
        file = files[0]
        file_id = file['id']

        request = drive_service.files().get_media(fileId=file_id)
        with io.FileIO(save_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        
        print(f"File '{file_name}' downloaded successfully to '{save_path}'.")
        return True
    except Exception as e:
        print(f"An error occurred while downloading '{file_name}': {e}")
        return False

def upload_file(file_name, folder_id, local_path):
    """Upload a specific file to Google Drive."""
    print('this is upload')

    try:
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
        media = MediaFileUpload(local_path, resumable=True)
        print('this is upload')
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        print(f"File '{file_name}' uploaded successfully. File ID: {file.get('id')}")
    except Exception as e:
        print(f"An error occurred while uploading '{file_name}': {e}")

# Function to generate the local label encoder
def create_local_label_encoder(local_data, local_encoder_path):
    """
    Creates and saves a label encoder for the client's dataset.

    Args:
        local_data: The local dataset (Pandas DataFrame).
        local_encoder_path: Path to save the local label encoder (pickle file).
    """
    # Extract unique crops from the client's dataset
    unique_crops = local_data['CROP'].unique()

    # Create and fit the label encoder
    local_label_encoder = LabelEncoder()
    local_label_encoder.fit(unique_crops)

    # Save the local label encoder
    with open(local_encoder_path, "wb") as encoder_file:
        pickle.dump(local_label_encoder, encoder_file)

    print(f"Local label encoder saved to {local_encoder_path}.")
    return local_label_encoder

local_csv_path = "maharashtra_data.csv"  # Client's dataset
local_encoder_path = "local_label_encoder_mah.pkl"  # Path to save the local label encoder
global_model_path = "global_model.pkl"  # Global model sent by the server
updated_model_path = "updated_client_model_mah.pkl"  # Path to save the updated parameters
fine_tuned_global_model_path = "global_model.pkl"  #
global_label_encoder_path = "label_encoder.pkl"  # Path to the global label encoder

def train_model():
    # local_csv_path = "maharashtra_data.csv"  # Client's dataset
    # local_encoder_path = "local_label_encoder_guj.pkl"  # Path to save the local label encoder
    # global_model_path = "global_model.pkl"  # Global model sent by the server
    # updated_model_path = "updated_client_model_guj.pkl"  # Path to save the updated parameters
    # fine_tuned_global_model_path = "global_model.pkl"  # Path to save the fine-tuned global model

    
    # Load the local dataset
    local_data = pd.read_csv(local_csv_path)

    # Define features and target
    features = ['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL']
    target = 'CROP'

    # Drop unnecessary columns
    if 'STATE' in local_data.columns:
        local_data = local_data.drop(columns=['STATE'])

    # Step 1: Generate the local label encoder
    local_label_encoder = create_local_label_encoder(local_data, local_encoder_path)

    # Step 2: Load the global label encoder
    with open(global_label_encoder_path, "rb") as global_encoder_file:
        global_label_encoder = pickle.load(global_encoder_file)

    # Debug: Print global label encoder classes
    print(f"Global label encoder classes: {global_label_encoder.classes_}")

    # Step 3: Encode the target variable (CROP) using the global label encoder
    local_data[target] = global_label_encoder.transform(local_data[target])

    # Define the feature matrix (X) and target vector (y)
    X_local = local_data[features]
    y_local = local_data[target]

    # Step 4: Load the global model
    with open(global_model_path, "rb") as model_file:
        global_model = pickle.load(model_file)

    # Debug: Log details of the global model
    print(f"Loaded global model with coefficient shape: {global_model.coef_.shape}")

    # Step 5: Train the model locally on the client's dataset using global label mappings
    global_model.fit(X_local, y_local)

    # Step 6: Save the updated model parameters (fine-tuned weights)
    with open(updated_model_path, "wb") as updated_model_file:
        pickle.dump(global_model, updated_model_file)

    print(f"Updated client model parameters have been saved to {updated_model_path}.")

    # Step 7: Save the fine-tuned global model
    with open(fine_tuned_global_model_path, "wb") as fine_tuned_model_file:
        pickle.dump(global_model, fine_tuned_model_file)

    print(f"Fine-tuned global model has been saved to {fine_tuned_global_model_path}.")

if __name__ == "__main__":
    # Step 1: Download the global model
    if download_file(GLOBAL_MODEL_FILE, DOWNLOAD_FOLDER_ID, GLOBAL_MODEL_FILE):
        download_file(global_label_encoder_path, DOWNLOAD_FOLDER_ID, global_label_encoder_path)
        # Step 2: Train the model
        train_model()
            # Step 3: Upload the updated parameters
        upload_file(local_encoder_path, UPLOAD_FOLDER_ID, local_encoder_path)
        upload_file(updated_model_path, UPLOAD_FOLDER_ID, updated_model_path)