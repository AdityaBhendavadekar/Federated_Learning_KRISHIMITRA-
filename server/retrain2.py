import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload
import os
import io
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload


# Service account credentials file path
SERVICE_ACCOUNT_FILE = '/media/aditya/Work/Drive/edi-models-86ad22fb54fe.json'
SCOPES = ['https://www.googleapis.com/auth/drive']

# Folder ID of the main folder
MAIN_FOLDER_ID = '13Dcv7Er-M8b3FwOzRmswzWPkuyh17c_p'
FOLDER_UPLOAD_ID = '1IemFVUnKCwG21zriWhbwYZ63QFreuoeL'

# Authenticate and initialize the Drive API client
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

# Lists to store file paths
client_model_paths = []
client_label_encoder_paths = []

# Local directory to save the downloaded files
LOCAL_SAVE_DIR = '/'
os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

def list_files_in_folder(folder_id):
    """Recursively list all files in a folder and its subfolders."""
    files = []
    try:
        # Get files and folders in the current folder
        query = f"'{folder_id}' in parents"
        response = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        items = response.get('files', [])

        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':  # Check if it's a folder
                # Recursively list files in the subfolder
                files.extend(list_files_in_folder(item['id']))
            else:
                files.append(item)
        return files
    except Exception as e:
        print(f"Error listing files in folder {folder_id}: {e}")
        return files

def download_file(file_id, file_name):
    """Download a specific file from Google Drive."""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        save_path = os.path.join(file_name)
        with io.FileIO(save_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        print(f"File '{file_name}' downloaded successfully to '{save_path}'.")
        return save_path
    except Exception as e:
        print(f"Error downloading file '{file_name}': {e}")
        return None

def process_files_in_folder(client_model_paths, client_label_encoder_paths):
    """List and download files, categorizing them based on their names."""
    try:
        # List all files in the folder and subfolders
        files = list_files_in_folder(MAIN_FOLDER_ID)

        for file in files:
            file_name = file['name']
            file_id = file['id']

            if file_name.startswith('updated'):
                # Download file and add to the client_model_paths list
                downloaded_path = download_file(file_id, file_name)
                if downloaded_path:
                    client_model_paths.append(downloaded_path)

            elif file_name.startswith('local'):
                # Download file and add to the client_label_encoder_paths list
                downloaded_path = download_file(file_id, file_name)
                if downloaded_path:
                    client_label_encoder_paths.append(downloaded_path)

        print("Files categorized and downloaded successfully.")
        

    except Exception as e:
        print(f"Error processing files in folder: {e}")

# if __name__ == "__main__":


# Function to align the client model's coefficients to the global model
def align_model_to_global(client_model, global_model, global_label_encoder, client_label_encoder):
    """
    Aligns the coefficients and intercepts of a client model to match the global model.

    Args:
        client_model: The client's LogisticRegression model.
        global_model: The global LogisticRegression model.
        global_label_encoder: The global LabelEncoder used for class mappings.
        client_label_encoder: The client's LabelEncoder for local class mappings.

    Returns:
        Aligned coefficients and intercepts for the client model.
    """
    global_classes = global_label_encoder.classes_  # Global set of classes
    client_classes = client_label_encoder.classes_  # Client's set of classes

    # Debugging: Print global and client classes for verification
    print("\n[DEBUG] Global Encoder Classes:", global_classes)
    print("[DEBUG] Client Encoder Classes:", client_classes)

    # Create arrays to hold aligned coefficients and intercepts
    aligned_coef = np.zeros_like(global_model.coef_)
    aligned_intercept = np.zeros_like(global_model.intercept_)

    # Map client coefficients to the corresponding global indices
    for idx, crop in enumerate(client_classes):
        if crop in global_classes:
            global_idx = np.where(global_classes == crop)[0][0]
            aligned_coef[global_idx] = client_model.coef_[idx]
            aligned_intercept[global_idx] = client_model.intercept_[idx]
            # Debugging: Log alignment
            print(f"[DEBUG] Aligning client class '{crop}' (client idx {idx}) "
                  f"to global idx {global_idx}.")
        else:
            # Debugging: Log unmapped classes
            print(f"[WARNING] Client class '{crop}' not found in global classes!")

    return aligned_coef, aligned_intercept

# Function to aggregate model parameters from multiple clients
def aggregate_model_parameters(client_models, global_model, global_label_encoder, client_label_encoders):
    """
    Aggregates parameters from client models, aligning them to the global model structure.

    Args:
        client_models: List of client LogisticRegression models.
        global_model: The global LogisticRegression model.
        global_label_encoder: The global LabelEncoder for class mappings.
        client_label_encoders: List of client-specific LabelEncoders.

    Returns:
        A new global model with aggregated parameters.
    """
    num_clients = len(client_models)

    # Initialize averaged parameters
    avg_coef = np.zeros_like(global_model.coef_)
    avg_intercept = np.zeros_like(global_model.intercept_)

    # Align and sum parameters from all client models
    for client_idx, client_model in enumerate(client_models):
        print(f"\n[DEBUG] Processing Client Model {client_idx + 1}/{num_clients}")
        aligned_coef, aligned_intercept = align_model_to_global(
            client_model, global_model, global_label_encoder, client_label_encoders[client_idx]
        )
        avg_coef += aligned_coef
        avg_intercept += aligned_intercept

    # Average the parameters
    avg_coef /= num_clients
    avg_intercept /= num_clients

    # Update the global model with averaged parameters
    global_model.coef_ = avg_coef
    global_model.intercept_ = avg_intercept

    return global_model

# Function to update the global model using client models and label encoders
def update_global_model(global_model_path, client_model_paths, global_label_encoder_path, client_label_encoder_paths, updated_global_model_path):
    """
    Updates the global model by aggregating the parameters from client models.

    Args:
        global_model_path: Path to the global model.
        client_model_paths: List of paths to client models.
        global_label_encoder_path: Path to the global label encoder.
        client_label_encoder_paths: List of paths to client label encoders.
        updated_global_model_path: Path to save the updated global model.
    
    Returns:
        None
    """
    # Load the global model
    with open(global_model_path, "rb") as model_file:
        global_model = pickle.load(model_file)

    # Debugging: Log global model shape
    print(f"\n[DEBUG] Loaded Global Model with Coef Shape: {global_model.coef_.shape}")

    # Load the global label encoder
    with open(global_label_encoder_path, "rb") as le_file:
        global_label_encoder = pickle.load(le_file)

    # Debugging: Log global encoder classes
    print("[DEBUG] Global Label Encoder Classes:", global_label_encoder.classes_)

    # Load client models and their label encoders
    client_models = []
    client_label_encoders = []
    for path, encoder_path in zip(client_model_paths, client_label_encoder_paths):
        with open(path, "rb") as model_file:
            client_model = pickle.load(model_file)
            client_models.append(client_model)
            # Debugging: Log client model shape
            print(f"[DEBUG] Loaded Client Model from {path} with Coef Shape: {client_model.coef_.shape}")

        with open(encoder_path, "rb") as encoder_file:
            client_label_encoder = pickle.load(encoder_file)
            client_label_encoders.append(client_label_encoder)
            # Debugging: Log client label encoder classes
            print(f"[DEBUG] Client Label Encoder Classes from {encoder_path}:", client_label_encoder.classes_)

    # Aggregate the models
    new_global_model = aggregate_model_parameters(client_models, global_model, global_label_encoder, client_label_encoders)

    # Save the updated global model
    with open(updated_global_model_path, "wb") as model_file:
        pickle.dump(new_global_model, model_file)

    print(f"\n[DEBUG] Global model has been updated and saved to {updated_global_model_path}.")


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



# Example usage:
if __name__ == "__main__":
    # Paths to the necessary files
    client_model_paths = []  # Paths to client models
    client_label_encoder_paths = []  # Paths to client label encoders

    process_files_in_folder(client_model_paths, client_label_encoder_paths)

    print(f"Client Model Paths: {client_model_paths}")
    print(f"Client Label Encoder Paths: {client_label_encoder_paths}")

    global_model_path = "global_model.pkl"  # Path to the global model
    global_label_encoder_path = "label_encoder.pkl"  # Path to the global label encoder
    updated_global_model_path = "global_model.pkl"  # Path where the updated global model will be saved

    # Call the function to update the global model
    update_global_model(global_model_path, client_model_paths, global_label_encoder_path, client_label_encoder_paths, updated_global_model_path)
    upload_file(global_model_path, FOLDER_UPLOAD_ID,global_model_path)
    upload_file(global_label_encoder_path, FOLDER_UPLOAD_ID,global_label_encoder_path)
