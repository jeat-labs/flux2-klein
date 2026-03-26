# FLUX.2 Klein 4B — RunPod Serverless

Outpainting and inpainting powered by [FLUX.2 Klein Base 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) with the [fal outpaint LoRA](https://huggingface.co/fal/flux-2-klein-4B-outpaint-lora) on RunPod Serverless.

## Model

| Property | Value |
|---|---|
| Base Model | FLUX.2 Klein Base 4B (undistilled) |
| LoRA | fal/flux-2-klein-4B-outpaint-lora (scale 1.1) |
| Parameters | 4B |
| VRAM | ~13 GB (bfloat16) |
| Steps | 50 |
| Guidance Scale | 4.0 |
| License | Apache 2.0 |

## How It Works

FLUX.2 Klein has **no native mask-based inpainting**. Instead, the fal outpaint LoRA uses a **green-screen approach**:

1. Regions to fill are painted with pure green (`#00FF00`)
2. The LoRA interprets green as "generate content here"
3. The model seamlessly fills green regions while preserving original content

This works for both outpainting (green borders) and inpainting (green-masked regions).

## Outpainting API

```json
{
  "input": {
    "mode": "outpaint",
    "image_url": "https://example.com/photo.jpg",
    "target_width": 1024,
    "target_height": 1024,
    "steps": 50,
    "guidance_scale": 4.0,
    "seed": 42
  }
}
```

Or with explicit padding:

```json
{
  "input": {
    "mode": "outpaint",
    "image_base64": "<base64>",
    "pad_left": 200,
    "pad_right": 200,
    "pad_top": 0,
    "pad_bottom": 0
  }
}
```

## Inpainting API

```json
{
  "input": {
    "mode": "inpaint",
    "image_base64": "<base64>",
    "mask_base64": "<base64 L-mode, white=fill>",
    "prompt": "Fill the green spaces according to the image",
    "guidance_scale": 4.0
  }
}
```

The handler converts the binary mask to a green-screen image internally — you provide a standard white/black mask.

## Multi-Crop Mode

Generate one big square outpaint and get crops in all standard aspect ratios from a single render.

```json
{
  "input": {
    "mode": "multi_crop",
    "image_url": "https://example.com/product.jpg",
    "scale": 1.8,
    "ratios": ["1:1", "4:3", "16:9", "9:16", "4:5"]
  }
}
```

### Standard Ratios

| Ratio | Use Case |
|---|---|
| `1:1` | Instagram, Shopify square |
| `4:3` | Product listing standard |
| `3:4` | Portrait product |
| `16:9` | Hero banner, YouTube |
| `9:16` | Stories, Reels, TikTok |
| `4:5` | Instagram portrait |
| `5:4` | Landscape product |

## Parameters

| Parameter | Default | Range | Notes |
|---|---|---|---|
| `guidance_scale` | 4.0 | 1-10 | Base model default |
| `steps` | 50 | 20-100 | Undistilled model needs more steps |
| `seed` | 42 | any int | For reproducibility |
| `prompt` | `"Fill the green spaces..."` | any string | LoRA default prompt works well |

## Performance

| GPU | Time (1024x1024) | VRAM |
|---|---|---|
| A40 48GB | ~15-20s | ~13GB |
| A100 40GB | ~10-15s | ~13GB |
| L4 24GB | ~25-30s | ~13GB |

## Comparison with FLUX.1 Fill Dev

| | FLUX.2 Klein 4B | FLUX.1 Fill Dev 12B |
|---|---|---|
| VRAM | ~13 GB | ~32 GB |
| Speed | 50 steps (~15-20s) | 28 steps (~10-30s) |
| Approach | Green-screen LoRA | Native mask conditioning |
| License | Apache 2.0 | Non-Commercial |
| Quality | Very good | Excellent |
| GPU tier | L4/A40/A100 | A40/A100/H100 |

FLUX.2 Klein is the **cost-effective, commercially-licensed** alternative. Runs on cheaper GPUs with 60% less VRAM.

## Build & Deploy

```bash
docker build -t jeatlabs/flux2-klein:v1 .
docker push jeatlabs/flux2-klein:v1
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md)
