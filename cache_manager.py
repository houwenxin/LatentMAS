"""
Embedding Cache Manager for LatentMAS API

Manages server-side storage of latent embeddings and KV cache with TTL expiration.
Used for hybrid vLLM + HuggingFace mode where HF generates latent embeddings
and vLLM uses them for text generation.

Optimization: Stores both embeddings (for vLLM text generation) and KV cache
(for HF latent generation) to avoid reprocessing past context on each call.
"""

import gc
import time
import uuid
import threading
from typing import Dict, Optional, Tuple, Any, Union, List
import torch
try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = None
    DynamicCache = None


def _move_tensor_to_device(tensor: torch.Tensor, device: Union[str, torch.device]) -> torch.Tensor:
    """Move a tensor to specified device."""
    if tensor is None:
        return None
    return tensor.to(device)


def _move_kv_cache_to_device(
    past_kv: Optional[Tuple], 
    device: Union[str, torch.device]
) -> Optional[Tuple]:
    """Move KV cache tuple to specified device."""
    if past_kv is None:
        return None
    
    # Handle Cache objects (DynamicCache, etc.)
    if Cache is not None and isinstance(past_kv, Cache):
        # Convert to legacy format, move, then convert back
        legacy = past_kv.to_legacy_cache()
        moved_legacy = tuple(
            tuple(t.to(device) for t in layer)
            for layer in legacy
        )
        return past_kv.__class__.from_legacy_cache(moved_legacy)
    
    # Handle legacy tuple format: tuple of (key, value) tuples per layer
    return tuple(
        tuple(t.to(device) for t in layer)
        for layer in past_kv
    )


class EmbeddingCacheManager:
    """
    Thread-safe in-memory cache manager for latent embeddings and KV cache.
    
    Each session stores:
    - embeddings: The latent embedding tensor [1, L, H] (for vLLM text generation)
    - past_kv: HuggingFace KV cache tuple (for efficient latent generation)
    - shape: Shape of the embeddings for debugging
    - latent_steps: Number of latent steps taken
    - created_at: Timestamp for TTL management
    - last_accessed: Last access timestamp
    
    The KV cache is stored on CPU to save GPU memory and moved to GPU on access.
    This avoids reprocessing all past embeddings on each latent call.
    """
    
    def __init__(self, ttl_seconds: int = 1800, cleanup_interval: int = 1800):
        """
        Args:
            ttl_seconds: Time-to-live for cache entries (default 30 minutes)
            cleanup_interval: Interval for cleanup thread (default 30 minutes)
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def stop_cleanup_thread(self) -> None:
        """Stop background cleanup thread."""
        self._running = False
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=1.0)
    
    def _cleanup_loop(self) -> None:
        """Background loop to remove expired entries."""
        while self._running:
            time.sleep(self._cleanup_interval)
            self._cleanup_expired()
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        now = time.time()
        expired_keys = []
        
        with self._lock:
            for session_id, entry in self._cache.items():
                if now - entry["last_accessed"] > self._ttl:
                    expired_keys.append(session_id)
            
            for key in expired_keys:
                del self._cache[key]
        
        # Batch cleanup: release memory only during periodic cleanup, not per-operation
        if expired_keys:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return len(expired_keys)
    
    def create(
        self, 
        embeddings: torch.Tensor, 
        latent_steps: int,
        session_id: Optional[str] = None,
        past_kv: Optional[Tuple] = None,
    ) -> str:
        """
        Create a new cache entry for latent embeddings and KV cache.
        
        Args:
            embeddings: The latent embeddings tensor [1, L, H] (will be moved to CPU)
            latent_steps: Number of latent steps taken
            session_id: Optional custom session ID. If None, UUID is generated.
            past_kv: Optional HuggingFace KV cache tuple (will be moved to CPU)
            
        Returns:
            The session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Move embeddings and KV cache to CPU to save GPU memory
        embeddings_cpu = _move_tensor_to_device(embeddings, 'cpu')
        past_kv_cpu = _move_kv_cache_to_device(past_kv, 'cpu')
        shape = list(embeddings_cpu.shape) if embeddings_cpu is not None else None
        
        now = time.time()
        with self._lock:
            self._cache[session_id] = {
                "embeddings": embeddings_cpu,
                "past_kv": past_kv_cpu,
                "shape": shape,
                "latent_steps": latent_steps,
                "created_at": now,
                "last_accessed": now,
            }
        
        return session_id
    
    def get(
        self, 
        session_id: str, 
        device: Union[str, torch.device] = 'cuda:0'
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve embeddings and KV cache by session ID and migrate to target device.
        
        Updates last_accessed timestamp on access.
        
        Args:
            session_id: Session identifier
            device: Target device to move tensors to (default: 'cuda:0')
        
        Returns:
            Dict with 'embeddings', 'past_kv', 'shape', 'latent_steps' or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(session_id)
            if entry is None:
                return None
            
            # Check TTL
            if time.time() - entry["last_accessed"] > self._ttl:
                del self._cache[session_id]
                return None
            
            # Update access time
            entry["last_accessed"] = time.time()
            embeddings_cpu = entry["embeddings"]
            past_kv_cpu = entry["past_kv"]
            shape = entry["shape"]
            latent_steps = entry["latent_steps"]
        
        # Migrate from CPU to GPU (outside lock to avoid blocking other operations)
        embeddings_gpu = _move_tensor_to_device(embeddings_cpu, device)
        past_kv_gpu = _move_kv_cache_to_device(past_kv_cpu, device)
        
        return {
            "embeddings": embeddings_gpu,
            "past_kv": past_kv_gpu,
            "shape": shape,
            "latent_steps": latent_steps,
        }
    
    def update(
        self, 
        session_id: str, 
        embeddings: torch.Tensor,
        latent_steps: int,
        past_kv: Optional[Tuple] = None,
    ) -> bool:
        """
        Update existing cache entry.
        
        Args:
            embeddings: The embeddings tensor to store (will be moved to CPU)
            latent_steps: Number of latent steps
            past_kv: Optional HuggingFace KV cache tuple (will be moved to CPU)
        
        Returns:
            True if updated, False if session not found
        """
        # Move embeddings and KV cache to CPU to save GPU memory
        embeddings_cpu = _move_tensor_to_device(embeddings, 'cpu')
        past_kv_cpu = _move_kv_cache_to_device(past_kv, 'cpu')
        shape = list(embeddings_cpu.shape) if embeddings_cpu is not None else None
        
        with self._lock:
            if session_id not in self._cache:
                return False
            
            self._cache[session_id]["embeddings"] = embeddings_cpu
            self._cache[session_id]["past_kv"] = past_kv_cpu
            self._cache[session_id]["shape"] = shape
            self._cache[session_id]["latent_steps"] = latent_steps
            self._cache[session_id]["last_accessed"] = time.time()
            return True
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a cache entry.
        
        Returns:
            True if deleted, False if not found
        """
        deleted = False
        with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]
                deleted = True
        
        return deleted
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists and is not expired."""
        with self._lock:
            entry = self._cache.get(session_id)
            if entry is None:
                return False
            if time.time() - entry["last_accessed"] > self._ttl:
                del self._cache[session_id]
                return False
            return True
    
    def size(self) -> int:
        """Return number of cached sessions."""
        with self._lock:
            return len(self._cache)
    
    def clear(self) -> int:
        """Clear all cache entries. Returns count of cleared entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        
        return count


# Backwards compatibility alias
CacheManager = EmbeddingCacheManager


# Global cache manager instance
_cache_manager: Optional[EmbeddingCacheManager] = None


def get_cache_manager(ttl_seconds: int = 1800) -> EmbeddingCacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = EmbeddingCacheManager(ttl_seconds=ttl_seconds)
        _cache_manager.start_cleanup_thread()
    return _cache_manager
