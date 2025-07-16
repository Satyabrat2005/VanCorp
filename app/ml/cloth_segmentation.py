import torch
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from collections import OrderedDict

# Define CIHP cloth class labels
CIHP_CLOTH_LABELS = {
    5: 'UpperClothes',
    6: 'Dress',
    7: 'Coat'
}

# Load pretrained DeepLabV3+ MobileNetV3 model
MODEL_PATH = "assets/cihp/deep_lab_v3/deep_lab_v3.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model architecture (num_classes must match training config)
model = deeplabv3_mobilenet_v3_large(pretrained=False, num_classes=20).to(device)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Extract actual weights from checkpoint
if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
else:
    raise RuntimeError("Expected 'model_state' key in checkpoint")

# Remove 'module.' prefix (for DataParallel models)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[new_key] = v

# Load model weights (using strict=False to ignore unexpected keys)
model.load_state_dict(new_state_dict, strict=False)
model.eval()

# Define transform
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def segment_clothes(image: np.ndarray, keep_labels=[5, 6, 7]) -> np.ndarray:
    """
    Segments clothing regions from an image using DeepLabV3.

    Args:
        image (np.ndarray): Input BGR image.
        keep_labels (list): List of CIHP label indices to keep.

    Returns:
        np.ndarray: Binary-masked segmented clothing image.
    """
    original_size = (image.shape[1], image.shape[0])  # (W, H)
    input_tensor = transform(image).unsqueeze(0).to(device) # type: ignore

    with torch.no_grad():
        output = model(input_tensor)["out"]
        predicted = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Create binary mask for clothing labels
    mask = np.zeros_like(predicted, dtype=np.uint8)
    for label in keep_labels:
        mask[predicted == label] = 255

    # Resize mask back to original resolution
    resized_mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Apply mask to original image
    segmented_image = cv2.bitwise_and(image, image, mask=resized_mask)

    return segmented_image
