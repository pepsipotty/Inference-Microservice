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

| Variable | Default | Description |
|----------|---------|-------------|
| `FIREBASE_STORAGE_BUCKET` | `dpo-frontend.firebasestorage.app` | Firebase Storage bucket name |
| `GOOGLE_APPLICATION_CREDENTIALS` | `/app/serviceAccountKey.json` | Path to Firebase service account key |
| `BASE_MODEL_NAME` | `EleutherAI/pythia-2.8b` | HuggingFace model identifier |
| `MAX_NEW_TOKENS` | `75` | Maximum tokens to generate |
| `TEMPERATURE` | `0.2` | Sampling temperature (lower = more focused) |
| `TOP_P` | `0.85` | Nucleus sampling threshold |
| `REPETITION_PENALTY` | `1.3` | Penalty for repeating tokens (1.0 = no penalty) |
| `NO_REPEAT_NGRAM_SIZE` | `3` | Block repeating n-grams of this size |
| `DEVICE` | `cuda` | Device for inference (cuda/cpu) |

## Prompt Formatting

The service automatically reformats natural language questions into completion-style prompts optimized for small language models. This improves output quality by matching the model's training distribution.

**Automatic Transformations:**

- "What is X?" → "X is"
- "What are X?" → "X are"
- "Explain X" → "X is a concept that"
- "Tell me about X" → "X refers to"
- "How does X work?" → "X works by"
- "Why is X?" → "X is important because"
- "Describe X" → "X is"
- "Define X" → "X can be defined as"
- "Who is/was X?" → "X is/was"
- "Where is X?" → "X is located"
- "When did X?" → "X occurred"

All transformations are logged at INFO level for transparency.

## Notes

- First request will download model from Firebase (~30-60s depending on model size)
- Subsequent requests use cached models
- Fine-tuned models are loaded on-demand and unloaded after each request to save VRAM
