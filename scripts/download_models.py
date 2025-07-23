from huggingface_hub import hf_hub_download

def safe_download(repo_id, filename, local_dir):
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            force_download=True,        # Ensure a fresh, non-cached download
            resume_download=False       # Avoid using partial/corrupted files
        )
        print(f"Downloaded {filename} to {file_path}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    # Download TryOnGAN .pth
    safe_download(
        repo_id="Satyabrat266/tryon-models",
        filename="tryongan.pth",
        local_dir="assets/tryongan"
    )

    # Download TPS grid file
    safe_download(
        repo_id="Satyabrat266/tryon-models",
        filename="tps_grid.pth",
        local_dir="assets/tps"
    )
