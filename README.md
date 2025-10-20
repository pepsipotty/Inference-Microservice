# DPO Inference Microservice

FastAPI service for comparing base model (pythia-2.8b) and DPO fine-tuned model outputs.

## Requirements

- NVIDIA GPU with CUDA 11.8 (24GB+ VRAM recommended)
- Docker with NVIDIA Container Toolkit
- Firebase Storage account

## Setup

1. Get Firebase service account key from Firebase Console and save as `serviceAccountKey.json`

2. Build Docker image:
```bash
docker build -t dpo-inference:latest .
```

3. Run container:
```bash
docker run --gpus all \
  -p 8000:8000 \
  -v $(pwd)/serviceAccountKey.json:/app/serviceAccountKey.json \
  -v $(pwd)/model_cache:/app/model_cache \
  -e FIREBASE_STORAGE_BUCKET=dpo-frontend.firebasestorage.app \
  dpo-inference:latest
```

## API Usage

Health check:
```bash
curl http://localhost:8000/health
```

Run inference:
```bash
curl -X POST http://localhost:8000/inference/run \
  -H "Content-Type: application/json" \
  -d '{
    "modelId": "policy_model.pt",
    "prompt": "Your prompt here",
    "baseModel": "pythia_2_8b"
  }'
```

Response includes both base model and fine-tuned model outputs.

## Environment Variables

- `FIREBASE_STORAGE_BUCKET` - Firebase bucket name (default: dpo-frontend.firebasestorage.app)
- `MAX_NEW_TOKENS` - Max tokens to generate (default: 200)
- `TEMPERATURE` - Sampling temperature (default: 0.7)
- `TOP_P` - Nucleus sampling (default: 0.9)

## Notes

- First request will download model from Firebase (~30-60s depending on model size)
- Subsequent requests use cached models
- Fine-tuned models are loaded on-demand and unloaded after each request to save VRAM
