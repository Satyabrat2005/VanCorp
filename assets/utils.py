import torch
import torchvision.transforms as transforms
from PIL import Image
from assets.constatnt import MEAN, STD, IMG_HEIGHT, IMG_WIDTH

transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def image_to_tensor(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0) # type: ignore

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.squeeze(0).cpu().detach().clone()
    tensor = tensor * torch.tensor(STD).view(3,1,1) + torch.tensor(MEAN).view(3,1,1)
    tensor = torch.clamp(tensor, 0, 1)
    return transforms.ToPILImage()(tensor)