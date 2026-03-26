# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-03-25

### Added
- Outpainting via green-screen LoRA with target dimensions or explicit padding
- Inpainting via green-screen LoRA with binary mask input (white=fill, black=keep)
- Multi-crop mode: single square render with crops in 7 standard aspect ratios
- Image input via base64 or URL with automatic download
- FLUX.2 Klein Base 4B (undistilled) with fal outpaint LoRA fused at scale 1.1
- All inference wrapped in `torch.no_grad()` for optimal VRAM usage
- Per-request timing breakdown in response
- GPU metadata (name, VRAM) in response
