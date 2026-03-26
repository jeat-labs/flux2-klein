FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install --no-cache-dir \
    runpod \
    git+https://github.com/huggingface/diffusers \
    "transformers>=4.36.0" \
    accelerate \
    safetensors \
    sentencepiece \
    huggingface-hub \
    hf_transfer \
    httpx \
    Pillow \
    numpy

# Download FLUX.2 Klein Base 4B + outpaint LoRA
COPY src/download_models.py /tmp/download_models.py
RUN python3 /tmp/download_models.py && rm /tmp/download_models.py

COPY src/handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
