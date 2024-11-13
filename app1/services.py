import torch

# Paths to your saved model files
MTCNN_PATH = 'mtcnn_full_model.pth'
RESNET_PATH = 'inception_resnet_v1_full_model.pth'

# Global variables to store loaded models
mtcnn = None
resnet = None

def load_models():
    """Load MTCNN and ResNet models from saved files."""
    global mtcnn, resnet
    if mtcnn is None:
        print("Loading saved MTCNN model...")
        mtcnn = torch.load(MTCNN_PATH)  # Adjust if saved in different format
        print("MTCNN model loaded.")

    if resnet is None:
        print("Loading saved ResNet model...")
        resnet = torch.load(RESNET_PATH)  # Adjust if saved in different format
        resnet.eval()  # Set to evaluation mode if needed
        print("ResNet model loaded.")

def get_mtcnn():
    """Returns the loaded MTCNN model."""
    if mtcnn is None:
        load_models()
    return mtcnn

def get_resnet():
    """Returns the loaded ResNet model."""
    if resnet is None:
        load_models()
    return resnet
