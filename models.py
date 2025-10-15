from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any


class InferenceRequest(BaseModel):
    modelId: str = Field(..., description="Model ID in Firebase Storage")
    prompt: str = Field(..., description="Text prompt")
    baseModel: str = Field(default="pythia_2_8b")

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v

    @field_validator('modelId')
    @classmethod
    def validate_model_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Model ID cannot be empty")
        if not v.endswith('.pt'):
            raise ValueError("Model ID must end with .pt")
        return v


class InferenceResponse(BaseModel):
    requestId: str
    baseModelOutput: str
    fineTunedOutput: str
    modelId: str
    executionTimeMs: float
    timestamp: str


class ErrorResponse(BaseModel):
    requestId: str
    error: str
    code: str
    details: Optional[Dict[str, Any]] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    base_model_loaded: bool
    cuda_device: str
    vram_allocated_gb: float
