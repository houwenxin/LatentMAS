"""
LatentMAS OpenAI-Compatible API Server

Provides a /v1/chat/completions endpoint with three modes:
- normal: Standard text generation (no latent reasoning)
- latent: Generate latent representations, store KV cache in session
- text: Generate text using cached KV values from previous latent calls
"""

import argparse
import uuid
from typing import List, Optional, Literal
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import ModelWrapper
from cache_manager import get_cache_manager, CacheManager


# ==================== Request/Response Models ====================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request with LatentMAS extensions."""
    
    model: str = Field(default="latent-mas", description="Model identifier")
    messages: List[Message] = Field(..., description="Chat messages")
    
    # LatentMAS specific parameters
    mode: Literal["normal", "latent", "text"] = Field(
        default="normal",
        description="Operation mode: normal (standard generation), latent (generate & cache KV), text (use cached KV)"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for KV cache. Required for 'text' mode, optional for 'latent' mode"
    )
    latent_steps: Optional[int] = Field(
        default=None,
        description="Number of latent reasoning steps (only for 'latent' mode)"
    )

    debug_max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate for debug preview in latent mode"
    )
    debug_continuation_prompt: Optional[str] = Field(
        default=None,
        description="Continuation prompt for debug text generation in latent mode. If None, uses empty string."
    )

    # Standard generation parameters
    max_tokens: int = Field(default=256, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    
    # Optional: enable thinking token for reasoning models
    add_think_token: bool = Field(default=False, description="Add <think> token for reasoning models")


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # LatentMAS specific
    latent_steps: Optional[int] = Field(
        default=None,
        description="Number of latent reasoning steps taken (only for 'latent' mode)"
    )


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response with LatentMAS extensions."""
    
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[Choice]
    usage: Usage
    
    # LatentMAS specific
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for cached KV values (returned in latent/text modes)"
    )


# ==================== Global State ====================

model_wrapper: Optional[ModelWrapper] = None
cache_manager: Optional[CacheManager] = None
default_latent_steps: int = 10


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and cache manager on startup."""
    global model_wrapper, cache_manager, default_latent_steps
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LatentMAS API Server")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run model on")
    parser.add_argument("--latent_steps", type=int, default=10,
                        help="Default latent reasoning steps")
    parser.add_argument("--cache_ttl", type=int, default=1800,
                        help="Cache TTL in seconds (default 30 minutes)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    parser.add_argument("--latent_space_realign", action="store_true",
                        help="Enable latent space realignment")
    
    args, _ = parser.parse_known_args()
    
    default_latent_steps = args.latent_steps
    
    print(f"[API] Loading model: {args.model_name}")
    print(f"[API] Device: {args.device}")
    print(f"[API] Default latent steps: {args.latent_steps}")
    
    # Create a minimal args namespace for ModelWrapper
    model_args = argparse.Namespace(
        latent_space_realign=args.latent_space_realign,
        device=args.device,
        device2=args.device,  # Same device for HF model
        use_second_HF_model=False,
        enable_prefix_caching=False,
        method="latent_mas",
    )
    
    model_wrapper = ModelWrapper(
        model_name=args.model_name,
        device=torch.device(args.device),
        use_vllm=False,  # Use HF backend for API
        args=model_args,
    )
    
    cache_manager = get_cache_manager(ttl_seconds=args.cache_ttl)
    
    print(f"[API] Server ready!")
    
    yield
    
    # Cleanup
    cache_manager.stop_cleanup_thread()
    print("[API] Server shutting down")


# ==================== FastAPI App ====================

app = FastAPI(
    title="LatentMAS API",
    description="OpenAI-compatible API with latent multi-agent reasoning support",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Helper Functions ====================

def build_prompt_from_messages(messages: List[Message]) -> str:
    """Convert messages to chat prompt using model's chat template."""
    message_dicts = [{"role": m.role, "content": m.content} for m in messages]
    return model_wrapper.render_chat(message_dicts, add_generation_prompt=True)


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(model_wrapper.tokenizer.encode(text, add_special_tokens=False))


# ==================== Endpoints ====================

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with LatentMAS extensions.
    
    Modes:
    - normal: Standard text generation without latent reasoning
    - latent: Generate latent representations and cache KV values
    - text: Generate text using cached KV values from previous latent calls
    """
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Build prompt from messages
    prompt = build_prompt_from_messages(request.messages)
    prompt_tokens = count_tokens(prompt)
    
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    if request.mode == "normal":
        # Standard generation - no latent, no cache
        generated_text = model_wrapper.generate_text_for_api(
            prompt,
            past_key_values=None,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            add_think_token=request.add_think_token,
        )
        
        completion_tokens = count_tokens(generated_text)
        
        return ChatCompletionResponse(
            id=response_id,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=generated_text),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            session_id=None,
        )
    
    elif request.mode == "latent":
        # Generate latent representations and cache KV values
        # If latent_steps is None, generate until EOS token
        latent_steps = request.latent_steps  # Can be None for dynamic stopping
        
        # Load existing cache if session_id provided
        past_kv = None
        if request.session_id:
            past_kv = cache_manager.get(request.session_id)
            if past_kv is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session '{request.session_id}' not found or expired"
                )
        
        # Generate new latent representations with debug text output
        new_past_kv, debug_text, actual_steps, raw_token_ids = model_wrapper.generate_latent_for_api(
            prompt,
            latent_steps=latent_steps,
            past_key_values=past_kv,
            add_think_token=request.add_think_token,
            max_latent_steps=request.max_tokens,  # Use max_tokens as max latent steps
            debug_max_tokens=request.debug_max_tokens if request.debug_max_tokens is not None else 50,
            debug_continuation_prompt=request.debug_continuation_prompt,
        )
        
        # Store/update cache
        if request.session_id:
            cache_manager.update(request.session_id, new_past_kv)
            session_id = request.session_id
        else:
            session_id = cache_manager.create(new_past_kv)
        
        # Count debug text tokens
        debug_tokens = count_tokens(debug_text) if debug_text else 0
        
        # Build content message - now shows actual model continuation
        if latent_steps is not None:
            content = f"[Latent: {actual_steps} steps] {debug_text}"
        else:
            content = f"[Latent: dynamic {actual_steps} steps] {debug_text}"
        
        return ChatCompletionResponse(
            id=response_id,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role="assistant",
                        content=content,
                    ),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=actual_steps + debug_tokens,
                total_tokens=prompt_tokens + actual_steps + debug_tokens,
                latent_steps=actual_steps,
            ),
            session_id=session_id,
        )
    
    elif request.mode == "text":
        # Generate text using cached KV values
        if not request.session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required for 'text' mode"
            )
        
        past_kv = cache_manager.get(request.session_id)
        if past_kv is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found or expired"
            )
        
        generated_text = model_wrapper.generate_text_for_api(
            prompt,
            past_key_values=past_kv,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            add_think_token=request.add_think_token,
        )
        
        completion_tokens = count_tokens(generated_text)
        
        return ChatCompletionResponse(
            id=response_id,
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=generated_text),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            session_id=request.session_id,
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {request.mode}")


@app.get("/v1/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Check if a session exists."""
    if cache_manager.exists(session_id):
        return {"session_id": session_id, "exists": True}
    return {"session_id": session_id, "exists": False}


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its cached KV values."""
    deleted = cache_manager.delete(session_id)
    return {"session_id": session_id, "deleted": deleted}


@app.get("/v1/sessions")
async def list_sessions():
    """Get count of active sessions."""
    return {"active_sessions": cache_manager.size()}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_wrapper is not None,
        "active_sessions": cache_manager.size() if cache_manager else 0,
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="LatentMAS API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
