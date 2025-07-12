import torch
from assets.tryongan.tryon_gan import TryOnGenerator

def load_tryon_model(checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model = TryOnGenerator()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model
    