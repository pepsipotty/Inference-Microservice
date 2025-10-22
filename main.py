import os
import logging
import time
import uuid
import torch
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional

from models import InferenceRequest, InferenceResponse, ErrorResponse, HealthResponse
from firebase_client import FirebaseStorageClient, ModelNotFoundError, StorageConnectionError
from model_manager import BaseModelManager, FineTunedModelManager
import rag_retriever

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

base_model_manager: Optional[BaseModelManager] = None
finetuned_model_manager: Optional[FineTunedModelManager] = None
firebase_client: Optional[FirebaseStorageClient] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global base_model_manager, finetuned_model_manager, firebase_client

    try:
        logger.info("=" * 60)
        logger.info("Starting Inference Microservice")
        logger.info("=" * 60)

        firebase_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "dpo-frontend.firebasestorage.app")
        firebase_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/app/serviceAccountKey.json")
        base_model_name = os.getenv("BASE_MODEL_NAME", "EleutherAI/pythia-2.8b")
        device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"Total VRAM: {gpu_memory:.2f} GB")
        else:
            logger.warning("CUDA not available - running on CPU")

        logger.info(f"Initializing Firebase Storage client...")
        firebase_client = FirebaseStorageClient(
            bucket_name=firebase_bucket,
            credentials_path=firebase_creds
        )

        logger.info(f"Loading base model: {base_model_name}")
        base_model_manager = BaseModelManager(
            model_name=base_model_name,
            device=device
        )

        if not base_model_manager.is_loaded():
            raise RuntimeError("Failed to load base model")

        finetuned_model_manager = FineTunedModelManager(
            base_tokenizer=base_model_manager.tokenizer,
            device=device
        )

        logger.info("Initializing RAG retriever...")
        rag_enabled = rag_retriever.initialize()
        if rag_enabled:
            logger.info("RAG retriever initialized successfully")
        else:
            logger.warning("RAG disabled - PINECONE_API_KEY not set")

        logger.info("=" * 60)
        logger.info("Inference Microservice Ready")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    logger.info("Shutting down...")
    if finetuned_model_manager and finetuned_model_manager.is_loaded():
        finetuned_model_manager.unload()


app = FastAPI(
    title="DPO Inference Microservice",
    description="Side-by-side comparison of base and fine-tuned models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    logger.error(f"Unhandled exception for request {request_id}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "requestId": request_id,
            "error": "Internal server error",
            "code": "SERVER_ERROR",
            "details": {"message": str(exc)}
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        gpu_available = torch.cuda.is_available()
        base_model_loaded = base_model_manager is not None and base_model_manager.is_loaded()

        if gpu_available:
            cuda_device = torch.cuda.get_device_name(0)
            vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        else:
            cuda_device = "cpu"
            vram_allocated = 0.0

        return HealthResponse(
            status="ok" if base_model_loaded else "degraded",
            gpu_available=gpu_available,
            base_model_loaded=base_model_loaded,
            cuda_device=cuda_device,
            vram_allocated_gb=round(vram_allocated, 2)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            gpu_available=False,
            base_model_loaded=False,
            cuda_device="unknown",
            vram_allocated_gb=0.0
        )


@app.post("/inference/run", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    request_id = x_request_id or str(uuid.uuid4())

    try:
        start_time = time.time()
        logger.info(f"Request {request_id}: modelId={request.modelId}")

        if base_model_manager is None or finetuned_model_manager is None or firebase_client is None:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    requestId=request_id,
                    error="Service not initialized",
                    code="SERVER_ERROR",
                    details={}
                ).model_dump()
            )

        max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "50"))
        temperature = float(os.getenv("TEMPERATURE", "0.1"))
        top_p = float(os.getenv("TOP_P", "0.85"))
        repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.5"))
        no_repeat_ngram_size = int(os.getenv("NO_REPEAT_NGRAM_SIZE", "3"))

        try:
            kb_name = request.modelId.replace('.pt', '').split('_')[1]
            logger.debug(f"Request {request_id}: Extracted KB name: {kb_name}")

            context_results = rag_retriever.search(
                query=request.prompt,
                namespace=kb_name,
                top_k=3
            )

            if context_results:
                formatted_context = ""
                for i, item in enumerate(context_results, 1):
                    formatted_context += f"[Source {i}] (Community votes: N/A)\n"
                    formatted_context += f"Q: {item['question']}\n"
                    formatted_context += f"A: {item['answer']}\n\n"

                qa_prompt = f"""Based on this knowledge:

{formatted_context}Question: {request.prompt}
Answer:"""
                logger.info(f"Request {request_id}: Retrieved {len(context_results)} context items from '{kb_name}'")
            else:
                qa_prompt = f"Question: {request.prompt}\nAnswer:"
                logger.debug(f"Request {request_id}: No context retrieved, using original prompt")

        except Exception as e:
            logger.warning(f"Request {request_id}: RAG failed - {e}")
            qa_prompt = f"Question: {request.prompt}\nAnswer:"

        # Run base model
        logger.info(f"Request {request_id}: Running base model...")
        base_output = base_model_manager.generate(
            prompt=qa_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        # Load and run fine-tuned model
        logger.info(f"Request {request_id}: Loading fine-tuned model...")
        try:
            model_path = firebase_client.download_model(request.modelId)
            finetuned_model_manager.load_model(model_path, request.modelId)
        except ModelNotFoundError as e:
            logger.error(f"Request {request_id}: Model not found - {e}")
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    requestId=request_id,
                    error=str(e),
                    code="MODEL_NOT_FOUND",
                    details={"modelId": request.modelId}
                ).model_dump()
            )
        except StorageConnectionError as e:
            logger.error(f"Request {request_id}: Storage error - {e}")
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    requestId=request_id,
                    error=str(e),
                    code="SERVER_ERROR",
                    details={"modelId": request.modelId}
                ).model_dump()
            )

        logger.info(f"Request {request_id}: Running fine-tuned model...")
        finetuned_output = finetuned_model_manager.generate(
            prompt=qa_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        finetuned_model_manager.unload()

        total_time = (time.time() - start_time) * 1000
        logger.info(f"Request {request_id}: Completed in {total_time:.2f}ms")

        return InferenceResponse(
            requestId=request_id,
            baseModelOutput=base_output,
            fineTunedOutput=finetuned_output,
            modelId=request.modelId,
            executionTimeMs=round(total_time, 2),
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        )

    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"Request {request_id}: CUDA OOM - {e}")
        if finetuned_model_manager:
            finetuned_model_manager.unload()
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                requestId=request_id,
                error="Out of GPU memory",
                code="SERVER_ERROR",
                details={"message": "GPU memory exhausted"}
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Request {request_id}: Failed - {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                requestId=request_id,
                error="Inference failed",
                code="SERVER_ERROR",
                details={"message": str(e)}
            ).model_dump()
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
