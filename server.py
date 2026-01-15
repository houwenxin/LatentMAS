"""
LatentMAS OpenAI-Compatible API Server

Provides a /v1/chat/completions endpoint with three modes:
- normal: Standard text generation (no latent reasoning)
- latent: Generate latent embeddings using HuggingFace, store in session
- text: Generate text using vLLM with cached latent embeddings

Hybrid Mode (vLLM + HuggingFace):
- HuggingFace generates latent embeddings during 'latent' mode
- vLLM uses those embeddings for fast text generation in 'text' mode
- Requires --use_vllm flag to enable hybrid mode
"""

import argparse
import os
import uuid
import itertools
from typing import List, Optional, Literal, Dict
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import ModelWrapper
from cache_manager import get_cache_manager, EmbeddingCacheManager

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


# ==================== Helper Functions ====================

def detect_attention_implementation(model_wrapper: ModelWrapper) -> str:
    """Detect which attention implementation is being used by the model.
    
    Returns one of: 'flash_attention_2', 'sdpa', 'eager', 'vllm', or 'unknown'
    """
    if model_wrapper.use_vllm and _HAS_VLLM:
        return 'vllm'
    
    # Check transformers model config
    model = getattr(model_wrapper, 'model', None)
    if model is None:
        model = getattr(model_wrapper, 'HF_model', None)
    
    if model is None:
        return 'unknown'
    
    # Try to get attention implementation from model config
    config = getattr(model, 'config', None)
    if config is not None:
        attn_impl = getattr(config, '_attn_implementation', None)
        if attn_impl is not None:
            return attn_impl
        
        # Some models store it differently
        attn_impl = getattr(config, 'attn_implementation', None)
        if attn_impl is not None:
            return attn_impl
    
    # Fallback: check if flash-attn is available and model supports it
    try:
        import flash_attn
        # If flash_attn is installed and no explicit implementation is set,
        # transformers will try to use it by default on supported models
        return 'flash_attention_2 (inferred)'
    except ImportError:
        pass
    
    # Default for transformers without flash-attn is usually SDPA or eager
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return 'sdpa (inferred)'
    
    return 'eager (fallback)'


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
    
    # Latent space realignment
    latent_space_realign: bool = Field(
        default=False,
        description="Enable latent space realignment for better quality latent representations (only for 'latent' mode)"
    )
    
    # Latent only mode
    latent_only: bool = Field(
        default=False,
        description="Skip debug text generation in latent mode, only generate and cache KV values"
    )
    
    # vLLM acceleration
    use_vllm: bool = Field(
        default=False,
        description="Force use of vLLM for text generation (only for 'text' mode without session_id, or 'normal' mode when vLLM is enabled)"
    )


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
    kv_cache_shape: Optional[List[int]] = Field(
        default=None,
        description="Shape of latent embeddings [batch, seq_len, hidden_dim] (only for 'latent' mode)"
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
        description="Session ID for cached embeddings (returned in latent/text modes)"
    )


# ==================== Global State ====================

model_wrappers: Dict[str, ModelWrapper] = {}  # device -> ModelWrapper
device_pool: Optional[itertools.cycle] = None  # Round-robin device selector
cache_manager: Optional[EmbeddingCacheManager] = None
default_latent_steps: int = 10

# vLLM engine for high-throughput normal mode (optional)
vllm_engine: Optional["LLM"] = None
vllm_enabled: bool = False


# ==================== Helper Functions ====================

def get_model_wrapper() -> ModelWrapper:
    """Get next available model wrapper using round-robin load balancing."""
    device = next(device_pool)
    return model_wrappers[device]


# ==================== Lifespan ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and cache manager on startup."""
    global model_wrappers, device_pool, cache_manager, default_latent_steps, vllm_engine, vllm_enabled
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LatentMAS API Server")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or path")
    parser.add_argument("--devices", type=str, default=None,
                        help="[DEPRECATED] Use --hf_devices instead. Comma-separated list of HF devices.")
    parser.add_argument("--hf_devices", type=str, default=None,
                        help="Comma-separated list of devices for HuggingFace model (e.g., 'cuda:0'). "
                             "Required for latent/text modes. Defaults to 'cuda:0'.")
    parser.add_argument("--vllm_devices", type=str, default=None,
                        help="Comma-separated GPU indices for vLLM (e.g., '1,2,3'). "
                             "These GPUs will be isolated from HF via CUDA_VISIBLE_DEVICES. "
                             "If not specified, vLLM uses all GPUs not in --hf_devices.")
    parser.add_argument("--latent_steps", type=int, default=None,
                        help="Default latent reasoning steps")
    parser.add_argument("--cache_ttl", type=int, default=1800,
                        help="Cache TTL in seconds (default 30 minutes)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    # vLLM options for hybrid mode
    parser.add_argument("--use_vllm", action="store_true",
                        help="Enable vLLM backend for 'normal' mode (high throughput). "
                             "Requires separate GPUs from HF model.")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization (0.0-1.0)")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=None,
                        help="vLLM tensor parallel size. Defaults to number of vLLM GPUs.")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="Maximum sequence length for vLLM. If not set, uses model's default. "
                             "Reduce this if you encounter KV cache memory errors.")
    
    args, _ = parser.parse_known_args()
    
    default_latent_steps = args.latent_steps
    
    # ==================== GPU Allocation Logic ====================
    # Determine available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[API] Detected {num_gpus} GPU(s)")
    
    # Handle deprecated --devices argument
    if args.devices and not args.hf_devices:
        print("[API] WARNING: --devices is deprecated, use --hf_devices instead")
        args.hf_devices = args.devices
    
    # Parse HF devices (default to cuda:0)
    if args.hf_devices:
        hf_devices = [d.strip() for d in args.hf_devices.split(',')]
    else:
        hf_devices = ["cuda:0"] if num_gpus > 0 else ["cpu"]
    
    # Extract GPU indices used by HF
    hf_gpu_indices = set()
    for dev in hf_devices:
        if dev.startswith("cuda:"):
            try:
                idx = int(dev.split(":")[1])
                hf_gpu_indices.add(idx)
            except (ValueError, IndexError):
                pass
    
    # Determine vLLM GPU indices
    if args.vllm_devices:
        vllm_gpu_indices = [int(x.strip()) for x in args.vllm_devices.split(',')]
    elif args.use_vllm and num_gpus > 1:
        # Auto-assign: all GPUs except those used by HF
        vllm_gpu_indices = [i for i in range(num_gpus) if i not in hf_gpu_indices]
    else:
        vllm_gpu_indices = []
    
    # Validate no overlap between HF and vLLM GPUs
    overlap = hf_gpu_indices.intersection(set(vllm_gpu_indices))
    if overlap and args.use_vllm:
        print(f"[API] ERROR: GPU overlap detected between HF and vLLM: {overlap}")
        print(f"[API] HF devices: {hf_devices} (GPU indices: {hf_gpu_indices})")
        print(f"[API] vLLM GPU indices: {vllm_gpu_indices}")
        raise ValueError(
            f"GPU conflict: indices {overlap} are used by both HF and vLLM. "
            f"Use --hf_devices and --vllm_devices to specify non-overlapping GPUs."
        )
    
    # Check if vLLM can be enabled
    if args.use_vllm and not vllm_gpu_indices:
        if num_gpus <= 1:
            print("[API] WARNING: --use_vllm requires multiple GPUs. Only 1 GPU available.")
            print("[API] Disabling vLLM, using HuggingFace only.")
            args.use_vllm = False
        else:
            print("[API] WARNING: No GPUs available for vLLM after HF allocation.")
            args.use_vllm = False
    
    print(f"[API] Loading model: {args.model_name}")
    print(f"[API] HuggingFace devices: {hf_devices}")
    if args.use_vllm:
        print(f"[API] vLLM GPU indices: {vllm_gpu_indices}")
    print(f"[API] Default latent steps: {args.latent_steps}")
    
    # ==================== Load HuggingFace Model First ====================
    # Load HF model BEFORE vLLM to ensure it claims its GPU memory first
    for device in hf_devices:
        print(f"[API] Loading HuggingFace model on {device}...")
        
        # Create a minimal args namespace for ModelWrapper
        model_args = argparse.Namespace(
            latent_space_realign=False,  # Per-request parameter, always build both matrices at startup
            device=device,
            device2=device,  # Same device for HF model
            use_second_HF_model=False,
            enable_prefix_caching=False,
            method="latent_mas",
        )
        
        model_wrappers[device] = ModelWrapper(
            model_name=args.model_name,
            device=torch.device(device),
            use_vllm=False,  # Use HF backend for API
            args=model_args,
        )
        
        # Detect and log attention implementation
        attn_impl = detect_attention_implementation(model_wrappers[device])
        print(f"[API] Attention implementation on {device}: {attn_impl}")
    
    # Create round-robin device selector for HF models
    device_pool = itertools.cycle(hf_devices)
    
    # ==================== Load vLLM Engine ====================
    # Load vLLM AFTER HF model, using isolated GPUs
    if args.use_vllm:
        if not _HAS_VLLM:
            print("[API] WARNING: --use_vllm specified but vLLM not installed. Falling back to HF.")
        elif vllm_gpu_indices:
            # Set CUDA_VISIBLE_DEVICES for vLLM subprocess/workers
            # vLLM will see these as cuda:0, cuda:1, etc.
            vllm_visible_devices = ",".join(str(i) for i in vllm_gpu_indices)
            
            # Determine tensor parallel size
            tensor_parallel_size = args.vllm_tensor_parallel_size or len(vllm_gpu_indices)
            
            print(f"[API] Loading vLLM engine for hybrid mode...")
            print(f"[API] vLLM CUDA_VISIBLE_DEVICES (for workers): {vllm_visible_devices}")
            print(f"[API] vLLM GPU memory utilization: {args.vllm_gpu_memory_utilization}")
            print(f"[API] vLLM tensor parallel size: {tensor_parallel_size}")
            
            # Store original CUDA_VISIBLE_DEVICES
            original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            
            try:
                # Temporarily set CUDA_VISIBLE_DEVICES for vLLM initialization
                os.environ["CUDA_VISIBLE_DEVICES"] = vllm_visible_devices
                
                vllm_engine = LLM(
                    model=args.model_name,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    max_model_len=args.vllm_max_model_len,
                    trust_remote_code=True,
                    enable_prompt_embeds=True,  # Required for hybrid mode with latent embeddings
                    enable_prefix_caching=True,  # Improve performance with cached KV
                )
                vllm_enabled = True
                print(f"[API] vLLM engine loaded successfully on GPUs: {vllm_gpu_indices}")
                print(f"[API] vLLM prompt_embeds enabled for hybrid latent mode")
            finally:
                # Restore original CUDA_VISIBLE_DEVICES
                if original_cuda_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
                elif "CUDA_VISIBLE_DEVICES" in os.environ:
                    del os.environ["CUDA_VISIBLE_DEVICES"]
    
    cache_manager = get_cache_manager(ttl_seconds=args.cache_ttl)
    
    print(f"[API] Server ready with {len(hf_devices)} HF device(s)!")
    if vllm_enabled:
        print(f"[API] Hybrid mode enabled:")
        print(f"[API]   - HuggingFace generates latent embeddings (latent mode)")
        print(f"[API]   - vLLM uses embeddings for text generation (text mode)")
    
    yield
    
    # Cleanup
    cache_manager.stop_cleanup_thread()
    if vllm_engine is not None:
        del vllm_engine
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
    return get_model_wrapper().render_chat(message_dicts, add_generation_prompt=True)


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(get_model_wrapper().tokenizer.encode(text, add_special_tokens=False))


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
    if not model_wrappers:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Build prompt from messages
    prompt = build_prompt_from_messages(request.messages)
    prompt_tokens = count_tokens(prompt)
    
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    if request.mode == "normal":
        # Standard generation - no latent, no cache
        # Use vLLM if enabled for high throughput, otherwise fall back to HF
        if vllm_enabled and vllm_engine is not None:
            # vLLM path: high throughput text generation
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )
            # Handle think token for vLLM
            vllm_prompt = f"{prompt}<think>" if request.add_think_token else prompt
            outputs = vllm_engine.generate([vllm_prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip() if outputs else ""
        else:
            # HuggingFace fallback
            model_wrapper = get_model_wrapper()
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
        # Generate latent representations and cache embeddings for hybrid vLLM mode
        latent_steps = request.latent_steps or default_latent_steps or 10
        
        # Hybrid mode requires vLLM to be enabled
        if not vllm_enabled or vllm_engine is None:
            raise HTTPException(
                status_code=400,
                detail="Latent mode requires vLLM to be enabled. Start server with --use_vllm flag."
            )
        
        model_wrapper = get_model_wrapper()
        
        # Load existing embeddings if session_id provided (for accumulating across calls)
        past_embeddings = None
        if request.session_id:
            cached_data = cache_manager.get(request.session_id, device=str(model_wrapper.device))
            if cached_data is not None:
                past_embeddings = cached_data["embeddings"]
        
        # Generate latent embeddings using HuggingFace
        latent_embeddings, actual_steps, _ = model_wrapper.generate_latent_with_embeddings_for_api(
            prompt,
            latent_steps=latent_steps,
            add_think_token=request.add_think_token,
            latent_space_realign=request.latent_space_realign,
            past_embeddings=past_embeddings,
            latent_only=request.latent_only,
        )
        
        # Store embeddings in cache
        if request.session_id:
            cache_manager.update(request.session_id, latent_embeddings, actual_steps)
            session_id = request.session_id
        else:
            session_id = cache_manager.create(latent_embeddings, actual_steps)
        
        # Build content message with debug info
        debug_text = ""
        if request.debug_max_tokens and request.debug_max_tokens > 0:
            generated_text = model_wrapper.generate_text_with_embeddings_for_api(
                prompt,
                latent_embeddings=past_embeddings,
                vllm_engine=vllm_engine,
                max_new_tokens=request.debug_max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                add_think_token=request.add_think_token,
            )
            debug_text = generated_text
        
        debug_tokens = count_tokens(debug_text) if debug_text else 0
        content = f"[Latent: {actual_steps} steps] {debug_text}"
        
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
                kv_cache_shape=list(latent_embeddings.shape),  # Now shows embedding shape [1, L, H]
            ),
            session_id=session_id,
        )
    
    elif request.mode == "text":
        # Generate text using cached latent embeddings with vLLM
        
        # Hybrid mode requires vLLM to be enabled
        if not vllm_enabled or vllm_engine is None:
            raise HTTPException(
                status_code=400,
                detail="Text mode requires vLLM to be enabled. Start server with --use_vllm flag."
            )
        
        # session_id is required for text mode (must have latent embeddings)
        if not request.session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required for 'text' mode. First call 'latent' mode to generate embeddings."
            )
        
        model_wrapper = get_model_wrapper()
        
        # Retrieve cached embeddings
        cached_data = cache_manager.get(request.session_id, device=str(model_wrapper.device))
        if cached_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found or expired"
            )
        
        latent_embeddings = cached_data["embeddings"]
        
        # Generate text using vLLM with injected latent embeddings
        generated_text = model_wrapper.generate_text_with_embeddings_for_api(
            prompt,
            latent_embeddings=latent_embeddings,
            vllm_engine=vllm_engine,
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
    """Delete a session and its cached embeddings."""
    deleted = cache_manager.delete(session_id)
    return {"session_id": session_id, "deleted": deleted}


@app.get("/v1/sessions")
async def list_sessions():
    """Get count of active sessions."""
    return {"active_sessions": cache_manager.size()}


@app.post("/v1/memory/cleanup")
async def cleanup_memory():
    """Force GPU memory cleanup. Useful for freeing cached CUDA memory."""
    import gc
    gc.collect()
    
    memory_stats = {}
    if torch.cuda.is_available():
        # Get memory stats before cleanup
        for device in model_wrappers.keys():
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            memory_stats[device] = {
                "before_allocated_mb": torch.cuda.memory_allocated(device_idx) / 1024 / 1024,
                "before_reserved_mb": torch.cuda.memory_reserved(device_idx) / 1024 / 1024,
            }
        
        torch.cuda.empty_cache()
        
        # Get memory stats after cleanup
        for device in model_wrappers.keys():
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            memory_stats[device]["after_allocated_mb"] = torch.cuda.memory_allocated(device_idx) / 1024 / 1024
            memory_stats[device]["after_reserved_mb"] = torch.cuda.memory_reserved(device_idx) / 1024 / 1024
            memory_stats[device]["freed_mb"] = (
                memory_stats[device]["before_reserved_mb"] - memory_stats[device]["after_reserved_mb"]
            )
    
    return {
        "status": "cleaned",
        "active_sessions": cache_manager.size() if cache_manager else 0,
        "memory_stats": memory_stats,
    }

@app.get("/v1/sessions/{session_id}/embeddings")
async def get_session_embeddings(session_id: str):
    """Retrieve cached embeddings for a session."""
    cached_data = cache_manager.get(session_id)
    if cached_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found or expired"
        )
    
    embeddings_shape = list(cached_data["embeddings"].shape)
    return {
        "session_id": session_id,
        "embeddings_shape": embeddings_shape,
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Get vLLM GPU info if available
    vllm_info = None
    if vllm_enabled and vllm_engine is not None:
        try:
            # vLLM doesn't expose GPU indices directly, but we can infer from config
            vllm_info = {
                "tensor_parallel_size": getattr(vllm_engine, 'llm_engine', {}) and 
                                        getattr(vllm_engine.llm_engine, 'parallel_config', None) and
                                        getattr(vllm_engine.llm_engine.parallel_config, 'tensor_parallel_size', 'unknown') or 'unknown',
            }
        except Exception:
            vllm_info = {"status": "running"}
    
    return {
        "status": "healthy",
        "model_loaded": len(model_wrappers) > 0,
        "hf_devices": list(model_wrappers.keys()),
        "active_sessions": cache_manager.size() if cache_manager else 0,
        "vllm_enabled": vllm_enabled,
        "vllm_info": vllm_info,
        "backends": {
            "normal_mode": "vllm" if vllm_enabled else "huggingface",
            "latent_mode": "huggingface (embeddings)" if vllm_enabled else "huggingface (requires --use_vllm)",
            "text_mode": "vllm (with cached embeddings)" if vllm_enabled else "requires --use_vllm",
        },
        "hybrid_mode": {
            "enabled": vllm_enabled,
            "description": "HF generates latent embeddings, vLLM uses them for text generation",
            "hf_gpus": list(model_wrappers.keys()),
            "vllm_isolated": vllm_enabled,
        },
    }


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="LatentMAS API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args, _ = parser.parse_known_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
