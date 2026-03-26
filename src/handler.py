"""RunPod Serverless handler for FLUX.2 Klein 4B outpainting/inpainting.

Model: FLUX.2 Klein Base 4B (undistilled) + fal outpaint LoRA
Architecture: FLUX transformer with green-screen LoRA for region filling
License: Apache 2.0

Outpainting/inpainting approach (green-screen LoRA):
  1. Place original image on pure green (#00FF00) canvas (outpaint)
     or paint masked regions with #00FF00 (inpaint)
  2. Pass green-marked image as reference to Flux2KleinPipeline
  3. LoRA interprets #00FF00 as "fill this region" and generates content
  4. guidance_scale=4.0 (base model), num_inference_steps=50
  5. LoRA fused at scale 1.1 (recommended by LoRA author)

Notes:
  - FLUX.2 Klein has NO native mask_image parameter — green-screen LoRA
    is the mechanism for spatial control
  - Uses the BASE (undistilled) model for LoRA compatibility
  - ~13GB VRAM in bfloat16, much lighter than FLUX.1 Fill Dev (32GB)
  - Green must be pure #00FF00 (RGB 0, 255, 0)
  - Green borders should be 5-25% of image dimension per side
  - All dimensions must be multiples of 16
"""

import base64
import io
import time

import numpy as np
import runpod
import torch
from PIL import Image, ImageDraw

from diffusers import Flux2KleinPipeline

# ============================================================
# Worker startup
# ============================================================

PIPE = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B",
    torch_dtype=torch.bfloat16,
    cache_dir="/model_cache",
).to("cuda")

PIPE.load_lora_weights(
    "fal/flux-2-klein-4B-outpaint-lora",
    weight_name="flux-outpaint-lora.safetensors",
    cache_dir="/model_cache",
)
PIPE.fuse_lora(lora_scale=1.1)

GPU_NAME = torch.cuda.get_device_name(0)
GPU_VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)


# ============================================================
# Helpers
# ============================================================


def _decode_image(job_input, key="image_base64", url_key="image_url"):
    if key in job_input:
        return Image.open(io.BytesIO(base64.b64decode(job_input[key]))).convert("RGB")
    if url_key in job_input:
        import httpx

        resp = httpx.get(job_input[url_key], timeout=30, follow_redirects=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return None


def _encode_image(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _round16(x):
    return (x // 16) * 16


def _apply_green_mask(image, mask):
    """Replace white regions in mask with pure green (#00FF00)."""
    img_arr = np.array(image)
    mask_arr = np.array(mask.resize(image.size))
    img_arr[mask_arr > 128] = [0, 255, 0]
    return Image.fromarray(img_arr)


# ============================================================
# Inpainting
# ============================================================

DEFAULT_PROMPT = "Fill the green spaces according to the image"


def _inpaint(job_input):
    image = _decode_image(job_input)
    if image is None:
        return {"error": "Provide image_base64 or image_url"}

    if "mask_base64" not in job_input:
        return {"error": "Provide mask_base64 (white=inpaint, black=keep)"}

    mask = Image.open(
        io.BytesIO(base64.b64decode(job_input["mask_base64"]))
    ).convert("L")

    green_image = _apply_green_mask(image, mask)

    prompt = job_input.get("prompt", DEFAULT_PROMPT)
    steps = job_input.get("steps", 50)
    guidance = job_input.get("guidance_scale", 4.0)
    seed = job_input.get("seed", 42)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = PIPE(
            prompt=prompt,
            image=green_image,
            height=_round16(image.height),
            width=_round16(image.width),
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
    t1 = time.perf_counter()

    return {
        "image_base64": _encode_image(result),
        "width": result.width,
        "height": result.height,
        "timings": {"total_ms": round((t1 - t0) * 1000, 1)},
    }


# ============================================================
# Outpainting
# ============================================================


def _outpaint(job_input):
    image = _decode_image(job_input)
    if image is None:
        return {"error": "Provide image_base64 or image_url"}

    orig_w, orig_h = image.size

    target_width = job_input.get("target_width")
    target_height = job_input.get("target_height")
    pad_left = job_input.get("pad_left", 0)
    pad_right = job_input.get("pad_right", 0)
    pad_top = job_input.get("pad_top", 0)
    pad_bottom = job_input.get("pad_bottom", 0)

    if target_width and target_height:
        total_pad_w = max(target_width - orig_w, 0)
        total_pad_h = max(target_height - orig_h, 0)
        pad_left = total_pad_w // 2
        pad_right = total_pad_w - pad_left
        pad_top = total_pad_h // 2
        pad_bottom = total_pad_h - pad_top

    if pad_left + pad_right + pad_top + pad_bottom == 0:
        return {"error": "Provide target_width/target_height or pad_left/right/top/bottom"}

    prompt = job_input.get("prompt", DEFAULT_PROMPT)
    steps = job_input.get("steps", 50)
    guidance = job_input.get("guidance_scale", 4.0)
    seed = job_input.get("seed", 42)

    new_w = _round16(orig_w + pad_left + pad_right)
    new_h = _round16(orig_h + pad_top + pad_bottom)

    # Green canvas with original centered
    canvas = Image.new("RGB", (new_w, new_h), (0, 255, 0))
    canvas.paste(image, (pad_left, pad_top))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.perf_counter()
    with torch.no_grad():
        result = PIPE(
            prompt=prompt,
            image=canvas,
            height=new_h,
            width=new_w,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
    t1 = time.perf_counter()

    return {
        "image_base64": _encode_image(result),
        "original_width": orig_w,
        "original_height": orig_h,
        "width": result.width,
        "height": result.height,
        "timings": {"total_ms": round((t1 - t0) * 1000, 1)},
    }


# ============================================================
# Multi-crop: one big square render, multiple aspect ratio crops
# ============================================================

STANDARD_CROPS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "4:5": (4, 5),
    "5:4": (5, 4),
}


def _multi_crop(job_input):
    """Generate one big square outpaint and return crops in multiple aspect ratios."""
    image = _decode_image(job_input)
    if image is None:
        return {"error": "Provide image_base64 or image_url"}

    orig_w, orig_h = image.size
    scale = job_input.get("scale", 1.8)
    requested = job_input.get("ratios", list(STANDARD_CROPS.keys()))
    prompt = job_input.get("prompt", DEFAULT_PROMPT)
    steps = job_input.get("steps", 50)
    guidance = job_input.get("guidance_scale", 4.0)
    seed = job_input.get("seed", 42)

    side = _round16(int(max(orig_w, orig_h) * scale))
    px = (side - orig_w) // 2
    py = (side - orig_h) // 2

    canvas = Image.new("RGB", (side, side), (0, 255, 0))
    canvas.paste(image, (px, py))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.perf_counter()
    with torch.no_grad():
        big_square = PIPE(
            prompt=prompt,
            image=canvas,
            height=side,
            width=side,
            guidance_scale=guidance,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
    t1 = time.perf_counter()

    cw, ch = big_square.size
    crops = {}
    for ratio_name in requested:
        if ratio_name not in STANDARD_CROPS:
            continue
        rw, rh = STANDARD_CROPS[ratio_name]
        crop_w = cw
        crop_h = int(cw * rh / rw)
        if crop_h > ch:
            crop_h = ch
            crop_w = int(ch * rw / rh)
        left = (cw - crop_w) // 2
        top = (ch - crop_h) // 2
        cropped = big_square.crop((left, top, left + crop_w, top + crop_h))
        crops[ratio_name] = _encode_image(cropped)

    return {
        "big_square": _encode_image(big_square),
        "big_square_size": side,
        "crops": crops,
        "ratios_generated": list(crops.keys()),
        "original_width": orig_w,
        "original_height": orig_h,
        "timings": {"total_ms": round((t1 - t0) * 1000, 1)},
    }


# ============================================================
# Handler
# ============================================================

MODES = {
    "inpaint": _inpaint,
    "outpaint": _outpaint,
    "multi_crop": _multi_crop,
}


def handler(job):
    job_input = job.get("input", {})
    mode = job_input.get("mode", "outpaint")

    if mode not in MODES:
        return {"error": f"Invalid mode '{mode}'. Use: {list(MODES.keys())}"}

    result = MODES[mode](job_input)
    result["mode"] = mode
    result["gpu"] = GPU_NAME
    result["gpu_vram_gb"] = GPU_VRAM_GB
    return result


runpod.serverless.start({"handler": handler})
