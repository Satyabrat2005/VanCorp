import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict
from diffusers import StableDiffusionInpaintPipeline  # type: ignore
from huggingface_hub import login, HfFolder

from app.ml.cloth_wrapping import wrap_cloth
from assets.tryongan.tryon_gan import TryOnGenerator

# ---------- Setup Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Hugging Face Token Login ----------
if not HfFolder.get_token():
    login()
HF_TOKEN = HfFolder.get_token()

# ---------- Load Stable Diffusion Inpainting ----------
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        use_safetensors=True,
        revision="fp16",
        token=HF_TOKEN,
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))  # Disable NSFW checker
except Exception as e:
    print(f"⚠️ Could not load Stable Diffusion pipeline. Reason: {e}")
    pipe = None

# ---------- Load TryOnGAN ----------
gan_generator = TryOnGenerator()
gan_generator.load_state_dict(torch.load("assets/tryongan/tryongan.pth", map_location=device))
gan_generator.eval().to(device)

# ---------- Fusion Function ----------
def fuse_tryon_output(
    user_image: np.ndarray,
    cloth_image: np.ndarray,
    keypoints: Dict[str, tuple],
    use_diffusers: bool = True
) -> np.ndarray:
    """
    Fuse cloth onto user image using TPS warping and refine with Stable Diffusion or TryOnGAN.
    """
    # Step 1: Warp cloth using TPS
    warped_cloth = wrap_cloth(cloth_image, keypoints, user_image.shape)  # type: ignore

    # Step 2: Binary mask creation
    gray = cv2.cvtColor(warped_cloth, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    mask_rgb = cv2.merge([mask, mask, mask])
    mask_inv_rgb = cv2.merge([mask_inv, mask_inv, mask_inv])

    user_bg = cv2.bitwise_and(user_image, mask_inv_rgb)
    cloth_fg = cv2.bitwise_and(warped_cloth, mask_rgb)
    blended = cv2.add(user_bg, cloth_fg)

    # Step 3: Use Stable Diffusion (if enabled and available)
    if use_diffusers and pipe is not None:
        pil_image = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)).resize((512, 512))
        pil_mask = Image.fromarray(mask).resize((512, 512))
        prompt = "a person wearing stylish modern clothes, professional photography"

        try:
            result = pipe(prompt=prompt, image=pil_image, mask_image=pil_mask).images[0]  # type: ignore
            final = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            return final
        except Exception as e:
            print(f"⚠️ Stable Diffusion failed: {e} — using TryOnGAN fallback.")

    # Step 4: Fallback to TryOnGAN
    input_tensor = torch.tensor(blended.transpose(2, 0, 1)).unsqueeze(0).float() / 127.5 - 1
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output_tensor = gan_generator(input_tensor)
        output_tensor = (output_tensor + 1) * 127.5
        output_image = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        final = np.clip(output_image, 0, 255).astype(np.uint8)

    return final
