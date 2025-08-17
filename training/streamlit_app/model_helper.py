import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ------------------------
# CONFIGURATION
# ------------------------
# Set the path to your pre-trained model.
# Adjust this path based on where 'saved_model.pth' is located relative to model_helper.py
# If 'model_helper.py' is in 'streamlit_app/' and 'saved_model.pth' is in 'streamlit_app/model/',
# then the path should be "model/saved_model.pth".
MODEL_PATH = "training/streamlit_app/model/saved_model.pth" # Corrected path

# Determine the device to run the model on (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the class names for your damage categories.
# Ensure these match the labels your model was trained on.
CLASS_NAMES = ['Front Breakage', 'Front Crushed', 'Front Normal',
               'Rear Breakage', 'Rear Crushed', 'Rear Normal']

# Define the image transformations required by your model.
# These should match the transformations used during model training.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 pixels
    transforms.ToTensor(),          # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize image
])

# ------------------------
# MODEL DEFINITION
# ------------------------
def build_model(num_classes=len(CLASS_NAMES)):
    """
    Builds and returns a pre-trained ResNet50 model adapted for your specific number of classes.
    This function should mirror the model architecture used in your 'model.ipynb' for training.
    """
    # Changed from ResNet18 to ResNet50, as suggested by the error messages' structure mismatch.
    # Using ResNet50_Weights.DEFAULT for most up-to-date weights.
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Replace the final fully connected layer to match the number of output classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ------------------------
# MODEL LOADING
# ------------------------
def load_model():
    """
    Loads the pre-trained model from the specified MODEL_PATH.
    It handles potential file not found errors and ensures the model is loaded to the correct device.
    It also adjusts state_dict keys if they have an unexpected prefix (e.g., 'model.', 'fc.1.').
    """
    model = build_model(num_classes=len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        try:
            # Load the state_dict
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

            # Create a new state_dict without unexpected prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'model.' prefix if it exists (e.g., 'model.conv1.weight' -> 'conv1.weight')
                if k.startswith('model.'):
                    k = k[6:]
                # Handle cases where the final 'fc' layer might be saved as 'fc.1.weight'
                # and rename it to 'fc.weight' to match the standard ResNet's fc layer name.
                if k.startswith('fc.1.'):
                    new_state_dict['fc.' + k[5:]] = v
                else:
                    new_state_dict[k] = v

            # Load the adjusted state_dict into the model.
            # Using strict=False will ignore any remaining keys that don't match,
            # which is useful if minor architectural differences (e.g., batch norm tracked buffers)
            # or if only part of the model was saved/loaded, ensuring the backbone loads.
            model.load_state_dict(new_state_dict, strict=False)
            model.to(DEVICE) # Move the model to the specified device
            model.eval() # Set the model to evaluation mode (important for inference)
            print(f"✅ Loaded model from {MODEL_PATH}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"❌ Model file not found at {MODEL_PATH}")
        return None

# Load the model globally when the script is imported
MODEL = load_model()

# ------------------------
# PREDICTION FUNCTION
# ------------------------
def predict(image_path, return_probabilities=False):
    """
    Predicts the damage class of an image using the loaded model.

    Args:
        image_path (str): Path to the image file to be predicted.
        return_probabilities (bool): If True, returns the class probabilities along with the predicted class.

    Returns:
        predicted_class (str): The name of the predicted damage class.
        probabilities (list of floats, optional): List of probabilities for each class, if return_probabilities is True.
        error_message (str): An error message if something goes wrong, otherwise None.
    """
    if MODEL is None:
        return "Error: Model not loaded.", None

    if not os.path.exists(image_path):
        return "Error: Image file not found.", None

    try:
        # Open and convert the image to RGB format
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return f"Error opening image: {e}", None

    # Apply transformations and add a batch dimension
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): # Disable gradient calculation for inference
        outputs = MODEL(image)
        # Apply softmax to get probabilities
        probs = F.softmax(outputs, dim=1)
        # Get the class with the highest probability
        conf, pred = torch.max(probs, 1)

    # Get the predicted class name
    predicted_class = CLASS_NAMES[pred.item()]
    # Convert probabilities to a list of floats
    probabilities = probs.squeeze().cpu().numpy().tolist()

    if return_probabilities:
        return predicted_class, probabilities
    else:
        return predicted_class, None

# ------------------------
# DEBUG / TEST (for local testing of model_helper.py)
# ------------------------
if __name__ == "__main__":
    # Example test: Replace 'test_image.jpg' with a real image path for testing
    # Make sure 'saved_model.pth' and a 'test_image.jpg' are in the same directory
    # as this model_helper.py script for this test to run.
    test_img = "temp_file.jpg" # This needs to be a valid image path for testing
    if os.path.exists(test_img):
        print(f"Testing prediction on {test_img}...")
        predicted_class, probs = predict(test_img, return_probabilities=True)

        if predicted_class.startswith("Error"):
            print(predicted_class)
        else:
            print(f"Predicted class: {predicted_class}")
            print(f"Probabilities: {probs}")
            for i, p in enumerate(probs):
                print(f"  {CLASS_NAMES[i]}: {p:.4f}")
    else:
        print(f"Please create a '{test_img}' file for local testing of model_helper.py")
