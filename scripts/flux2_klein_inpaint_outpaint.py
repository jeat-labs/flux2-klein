"""
FLUX.2 Klein 4B - Inpainting & Outpainting
============================================

KEY FACTS (verified from source code and model cards):
- Pipeline class: Flux2KleinPipeline (NOT FluxFillPipeline -- that is FLUX.1 Fill)
- FLUX.2 Klein does NOT have native mask-based inpainting. It is a unified
  text-to-image + reference-image-conditioned generation model.
- The `image` parameter accepts reference image(s) that CONDITION the generation
  (concatenated as latent tokens), but there is no `mask_image` parameter.
- For outpainting, the fal/flux-2-klein-4B-outpaint-lora uses a GREEN SCREEN
  approach: paint areas to fill as pure green (#00FF00), and the LoRA interprets
  that as "fill this region."

Models:
- black-forest-labs/FLUX.2-klein-4B       (distilled, 4 steps, guidance=1.0, Apache 2.0)
- black-forest-labs/FLUX.2-klein-base-4B  (undistilled, 50 steps, guidance=4.0, Apache 2.0)
- black-forest-labs/FLUX.2-klein-9B       (distilled, 4 steps, non-commercial)

Install:
    pip install git+https://github.com/huggingface/diffusers.git
    pip install transformers accelerate safetensors sentencepiece pillow
"""

import torch
from PIL import Image, ImageDraw
from diffusers import Flux2KleinPipeline


# =============================================================================
# 1. TEXT-TO-IMAGE (basic generation, no reference image)
# =============================================================================
def text_to_image():
    """Basic text-to-image generation with FLUX.2 Klein 4B (distilled)."""
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt="A serene mountain lake at sunset with reflections",
        height=1024,
        width=1024,
        guidance_scale=1.0,       # distilled model uses 1.0
        num_inference_steps=4,    # distilled model uses 4 steps
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("flux2_klein_t2i.png")
    print("Saved: flux2_klein_t2i.png")
    return image


# =============================================================================
# 2. REFERENCE-IMAGE-CONDITIONED GENERATION (image-to-image editing)
# =============================================================================
def reference_image_editing():
    """
    Pass one or more reference images to condition the generation.

    The pipeline concatenates VAE-encoded reference latents with the generation
    latents. This is NOT mask-based inpainting -- the entire output is generated
    while being conditioned on the reference(s).

    Use case: "Generate a new scene in the style of / containing elements from
    this reference image."
    """
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # Load reference image(s)
    ref_image = Image.open("my_reference.png").convert("RGB")

    image = pipe(
        prompt="A beautiful garden with the same house from the reference image",
        image=ref_image,                # single image or list of images
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("flux2_klein_edit.png")
    print("Saved: flux2_klein_edit.png")
    return image


# =============================================================================
# 3. MULTI-REFERENCE EDITING (combine multiple source images)
# =============================================================================
def multi_reference_editing():
    """Pass multiple reference images to blend concepts."""
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    ref1 = Image.open("subject.png").convert("RGB")
    ref2 = Image.open("background.png").convert("RGB")

    image = pipe(
        prompt="The person from the first image standing in the landscape from the second image",
        image=[ref1, ref2],           # list of reference images
        height=1024,
        width=1024,
        guidance_scale=1.0,
        num_inference_steps=4,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("flux2_klein_multi_ref.png")
    print("Saved: flux2_klein_multi_ref.png")
    return image


# =============================================================================
# 4. OUTPAINTING with Green-Screen LoRA (fal/flux-2-klein-4B-outpaint-lora)
# =============================================================================
def create_green_border_image(
    source_path: str,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """
    Place source image centered on a pure green (#00FF00) canvas.

    The LoRA is trained to interpret pure green (#00FF00) as "fill this area."
    Green border should be 5%-25% of the image dimension per side.
    Each side can have different widths.
    """
    source = Image.open(source_path).convert("RGB")

    # Create pure green canvas at target size
    canvas = Image.new("RGB", (target_width, target_height), (0, 255, 0))

    # Center the source image on the canvas
    paste_x = (target_width - source.width) // 2
    paste_y = (target_height - source.height) // 2
    canvas.paste(source, (paste_x, paste_y))

    return canvas


def outpaint_with_lora():
    """
    Outpainting using the fal green-screen LoRA.

    The LoRA (fal/flux-2-klein-4B-outpaint-lora) was trained on 1000 image
    pairs where green (#00FF00) borders mark areas to be filled. The LoRA
    teaches the model to seamlessly extend images into green regions.

    IMPORTANT: Uses the BASE (undistilled) model for LoRA compatibility.
    - guidance_scale=4.0
    - num_inference_steps=50
    """
    # Load the BASE (undistilled) model -- LoRAs are trained against the base
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    # Load the outpaint LoRA
    pipe.load_lora_weights(
        "fal/flux-2-klein-4B-outpaint-lora",
        weight_name="flux-outpaint-lora.safetensors",
    )
    # LoRA scale of 1.1 is recommended by the LoRA author
    pipe.fuse_lora(lora_scale=1.1)

    # Prepare image: place source on green canvas (extending all sides)
    # Example: extend a 768x768 image to 1024x1024
    green_image = create_green_border_image(
        source_path="my_photo.png",
        target_width=1024,
        target_height=1024,
    )
    green_image.save("debug_green_input.png")

    # The green-bordered image is passed as a reference image
    image = pipe(
        prompt="Fill the green spaces according to the image",
        image=green_image,
        height=1024,
        width=1024,
        guidance_scale=4.0,          # base model uses 4.0
        num_inference_steps=50,      # base model uses 50 steps
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("flux2_klein_outpaint.png")
    print("Saved: flux2_klein_outpaint.png")
    return image


# =============================================================================
# 5. INPAINTING via Green-Screen LoRA (mask a region with green)
# =============================================================================
def inpaint_with_green_mask():
    """
    Inpainting by painting the region to replace with pure green (#00FF00).

    Since FLUX.2 Klein has no native mask_image parameter, this LoRA-based
    approach is the way to do inpainting: paint over unwanted regions with
    pure green, and the LoRA fills them in.
    """
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    pipe.load_lora_weights(
        "fal/flux-2-klein-4B-outpaint-lora",
        weight_name="flux-outpaint-lora.safetensors",
    )
    pipe.fuse_lora(lora_scale=1.1)

    # Load source image and paint the region to inpaint with green
    source = Image.open("my_photo.png").convert("RGB")
    mask = Image.open("my_mask.png").convert("L")  # white = inpaint region

    # Apply green to masked region
    green_masked = apply_green_mask(source, mask)
    green_masked.save("debug_green_masked.png")

    image = pipe(
        prompt="Fill the green spaces according to the image, natural seamless result",
        image=green_masked,
        height=green_masked.height,
        width=green_masked.width,
        guidance_scale=4.0,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("flux2_klein_inpaint.png")
    print("Saved: flux2_klein_inpaint.png")
    return image


def apply_green_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Replace white regions in the mask with pure green (#00FF00) on the image.
    mask: grayscale PIL image, white (255) = region to fill.
    """
    import numpy as np

    img_arr = np.array(image)
    mask_arr = np.array(mask.resize(image.size))

    # Where mask is white (>128), replace with pure green
    fill_region = mask_arr > 128
    img_arr[fill_region] = [0, 255, 0]

    return Image.fromarray(img_arr)


# =============================================================================
# 6. DIRECTIONAL OUTPAINTING (extend in one direction only)
# =============================================================================
def outpaint_directional(
    source_path: str,
    direction: str = "right",
    extend_pixels: int = 256,
):
    """
    Extend an image in a specific direction.
    direction: "left", "right", "top", "bottom"
    extend_pixels: how many pixels to add (should be 5-25% of that dimension)
    """
    source = Image.open(source_path).convert("RGB")
    w, h = source.size

    if direction == "right":
        canvas = Image.new("RGB", (w + extend_pixels, h), (0, 255, 0))
        canvas.paste(source, (0, 0))
    elif direction == "left":
        canvas = Image.new("RGB", (w + extend_pixels, h), (0, 255, 0))
        canvas.paste(source, (extend_pixels, 0))
    elif direction == "bottom":
        canvas = Image.new("RGB", (w, h + extend_pixels), (0, 255, 0))
        canvas.paste(source, (0, 0))
    elif direction == "top":
        canvas = Image.new("RGB", (w, h + extend_pixels), (0, 255, 0))
        canvas.paste(source, (0, extend_pixels))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    # Ensure dimensions are divisible by 16 (vae_scale_factor * 2)
    new_w = (canvas.width // 16) * 16
    new_h = (canvas.height // 16) * 16
    canvas = canvas.resize((new_w, new_h), Image.LANCZOS)

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(
        "fal/flux-2-klein-4B-outpaint-lora",
        weight_name="flux-outpaint-lora.safetensors",
    )
    pipe.fuse_lora(lora_scale=1.1)

    image = pipe(
        prompt="Fill the green spaces according to the image, seamless natural extension",
        image=canvas,
        height=new_h,
        width=new_w,
        guidance_scale=4.0,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save(f"flux2_klein_outpaint_{direction}.png")
    print(f"Saved: flux2_klein_outpaint_{direction}.png")
    return image


# =============================================================================
# 7. ASPECT RATIO CONVERSION (e.g., portrait to landscape)
# =============================================================================
def convert_aspect_ratio(
    source_path: str,
    target_width: int = 1360,
    target_height: int = 768,
):
    """Convert portrait to landscape (or vice versa) via outpainting."""
    source = Image.open(source_path).convert("RGB")

    # Fit source within target while preserving aspect ratio
    source.thumbnail((target_width, target_height), Image.LANCZOS)

    # Center on green canvas
    canvas = Image.new("RGB", (target_width, target_height), (0, 255, 0))
    paste_x = (target_width - source.width) // 2
    paste_y = (target_height - source.height) // 2
    canvas.paste(source, (paste_x, paste_y))

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(
        "fal/flux-2-klein-4B-outpaint-lora",
        weight_name="flux-outpaint-lora.safetensors",
    )
    pipe.fuse_lora(lora_scale=1.1)

    # Ensure dimensions divisible by 16
    adj_w = (target_width // 16) * 16
    adj_h = (target_height // 16) * 16

    image = pipe(
        prompt="Fill the green spaces according to the image",
        image=canvas.resize((adj_w, adj_h), Image.LANCZOS),
        height=adj_h,
        width=adj_w,
        guidance_scale=4.0,
        num_inference_steps=50,
        generator=torch.Generator(device="cuda").manual_seed(42),
    ).images[0]

    image.save("flux2_klein_aspect_converted.png")
    print("Saved: flux2_klein_aspect_converted.png")
    return image


# =============================================================================
# PARAMETER REFERENCE
# =============================================================================
"""
Model Variant Comparison:
=========================
                        | FLUX.2-klein-4B     | FLUX.2-klein-base-4B  | FLUX.2-klein-9B
------------------------+---------------------+-----------------------+-------------------
Pipeline                | Flux2KleinPipeline  | Flux2KleinPipeline    | Flux2KleinPipeline
Distilled?              | Yes (4 steps)       | No (50 steps)         | Yes (4 steps)
guidance_scale          | 1.0                 | 4.0                   | 1.0
num_inference_steps     | 4                   | 50                    | 4
VRAM                    | ~13 GB              | ~13 GB                | ~29 GB
License                 | Apache 2.0          | Apache 2.0            | Non-commercial
LoRA compatible?        | Yes                 | Yes (preferred)       | Yes
Native mask inpainting? | NO                  | NO                    | NO

Inpainting/Outpainting Options:
================================
1. Green-screen LoRA (fal/flux-2-klein-4B-outpaint-lora):
   - Paint regions with #00FF00, LoRA fills them
   - Works for both inpainting and outpainting
   - Use with base (undistilled) model
   - LoRA scale: 1.1

2. Reference-image conditioning (built-in):
   - Pass image= parameter to condition generation
   - Entire image is regenerated, conditioned on reference
   - Not true inpainting (no masking)

Key constraints:
- Image dimensions must be divisible by 16
- Green borders should be 5-25% of image dimension
- Green must be pure #00FF00 (RGB 0,255,0)
"""


if __name__ == "__main__":
    # Uncomment the function you want to run:

    # text_to_image()
    # reference_image_editing()
    # multi_reference_editing()
    # outpaint_with_lora()
    # inpaint_with_green_mask()
    # outpaint_directional("my_photo.png", direction="right", extend_pixels=256)
    # convert_aspect_ratio("my_portrait.png", target_width=1360, target_height=768)

    print("FLUX.2 Klein 4B inpainting/outpainting script ready.")
    print("Uncomment a function in __main__ to run it.")
