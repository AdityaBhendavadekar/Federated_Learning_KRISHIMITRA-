import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Function to align the client model's coefficients to the global model
def align_model_to_global(client_model, global_model, label_encoder):
    """
    Aligns the coefficients and intercepts of a client model to match the global model.
    
    Args:
        client_model: The client's LogisticRegression model.
        global_model: The global LogisticRegression model.
        label_encoder: The global LabelEncoder used for class mappings.

    Returns:
        Aligned coefficients and intercepts for the client model.
    """
    global_classes = label_encoder.classes_  # Global set of classes
    client_classes = label_encoder.inverse_transform(range(len(client_model.coef_)))  # Client's classes

    # Create an array to hold aligned coefficients and intercepts
    aligned_coef = np.zeros_like(global_model.coef_)
    aligned_intercept = np.zeros_like(global_model.intercept_)

    # Map client coefficients to the corresponding global indices
    for idx, crop in enumerate(client_classes):
        if crop in global_classes:
            global_idx = np.where(global_classes == crop)[0][0]
            aligned_coef[global_idx] = client_model.coef_[idx]
            aligned_intercept[global_idx] = client_model.intercept_[idx]

    return aligned_coef, aligned_intercept

# Function to aggregate the model parameters from multiple clients
def aggregate_model_parameters(client_models, global_model, label_encoder):
    """
    Aggregates parameters from client models, aligning them to the global model structure.

    Args:
        client_models: List of client LogisticRegression models.
        global_model: The global LogisticRegression model.
        label_encoder: The global LabelEncoder for class mappings.

    Returns:
        A new global model with aggregated parameters.
    """
    num_clients = len(client_models)

    # Initialize averaged parameters
    avg_coef = np.zeros_like(global_model.coef_)
    avg_intercept = np.zeros_like(global_model.intercept_)

    # Align and sum parameters from all client models
    for client_model in client_models:
        aligned_coef, aligned_intercept = align_model_to_global(client_model, global_model, label_encoder)
        avg_coef += aligned_coef
        avg_intercept += aligned_intercept

    # Average the parameters
    avg_coef /= num_clients
    avg_intercept /= num_clients

    # Update the global model with averaged parameters
    global_model.coef_ = avg_coef
    global_model.intercept_ = avg_intercept

    return global_model

# Function to update the global model using client models and label encoder
def update_global_model(global_model_path, client_model_paths, label_encoder_path, updated_global_model_path):
    """
    Updates the global model by aggregating the parameters from the client models.

    Args:
        global_model_path: Path to the global model.
        client_model_paths: List of paths to client models.
        label_encoder_path: Path to the global label encoder.
        updated_global_model_path: Path to save the updated global model.
    
    Returns:
        None
    """
    # Load the global model
    with open(global_model_path, "rb") as model_file:
        global_model = pickle.load(model_file)

    # Load the global label encoder
    with open(label_encoder_path, "rb") as le_file:
        global_label_encoder = pickle.load(le_file)

    # Load client models
    client_models = []
    for path in client_model_paths:
        with open(path, "rb") as updated_model_file:
            client_model = pickle.load(updated_model_file)
            client_models.append(client_model)

    # Aggregate the models
    new_global_model = aggregate_model_parameters(client_models, global_model, global_label_encoder)

    # Save the updated global model
    with open(updated_global_model_path, "wb") as model_file:
        pickle.dump(new_global_model, model_file)

    print(f"Global model has been updated and saved to {updated_global_model_path}.")

# Example usage:
if __name__ == "__main__":
    # Paths to the necessary files
    global_model_path = "global_model.pkl"  # Path to the global model
    client_model_paths = ["updated_client_model_guj.pkl" ]  # Paths to client models
    label_encoder_path = "label_encoder.pkl"  # Path to the global label encoder
    updated_global_model_path = "updated_global_model.pkl"  # Path where the updated global model will be saved

    # Call the function to update the global model
    update_global_model(global_model_path, client_model_paths, label_encoder_path, updated_global_model_path)