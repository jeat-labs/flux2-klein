"""Download FLUX.2 Klein Base 4B and outpaint LoRA for Docker build."""
from huggingface_hub import hf_hub_download, list_repo_files

cache = "/model_cache"

# FLUX.2 Klein Base 4B (undistilled — required for LoRA compatibility)
repo = "black-forest-labs/FLUX.2-klein-base-4B"
print(f"Downloading {repo}...")
for f in list_repo_files(repo):
    print(f"  {f}")
    hf_hub_download(repo, f, cache_dir=cache)
print("Base model done")

# fal outpaint LoRA (green-screen approach)
print("Downloading outpaint LoRA...")
hf_hub_download(
    "fal/flux-2-klein-4B-outpaint-lora",
    filename="flux-outpaint-lora.safetensors",
    cache_dir=cache,
)
print("LoRA done")
