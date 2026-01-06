"""
KV Cache Manager for LatentMAS API

Manages server-side storage of KV caches with TTL expiration.
"""

import time
import uuid
import threading
from typing import Dict, Optional, Tuple, Any, Union
import torch
try:
    from transformers.cache_utils import Cache, DynamicCache
except ImportError:
    Cache = None
    DynamicCache = None

def _move_kv_to_device(past_key_values: Tuple, device: Union[str, torch.device]) -> Tuple:
    """
    Move KV cache tensors to specified device.
    Handles both Cache objects (DynamicCache) and legacy tuple format.
    
    Args:
        past_key_values: Cache object or nested tuple structure
        device: Target device (e.g., 'cpu', 'cuda:0')
        
    Returns:
        KV cache with all tensors moved to target device (always as legacy tuple format for storage)
    """
    if past_key_values is None:
        return None
    
    # Handle Cache objects (DynamicCache, etc.)
    if Cache is not None and isinstance(past_key_values, Cache):
        # Convert to legacy format, move tensors, keep as legacy for CPU storage
        legacy = past_key_values.to_legacy_cache()
        return tuple(
            tuple(tensor.to(device) for tensor in layer)
            for layer in legacy
        )
    
    # Handle legacy tuple format
    return tuple(
        tuple(tensor.to(device) for tensor in layer)
        for layer in past_key_values
    )


class CacheManager:
    """
    Thread-safe in-memory cache manager for KV caches.
    
    Each session stores:
    - past_key_values: The accumulated KV cache
    - created_at: Timestamp for TTL management
    - last_accessed: Last access timestamp
    """
    
    def __init__(self, ttl_seconds: int = 1800, cleanup_interval: int = 300):
        """
        Args:
            ttl_seconds: Time-to-live for cache entries (default 30 minutes)
            cleanup_interval: Interval for cleanup thread (default 5 minutes)
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
        
        return len(expired_keys)
    
    def create(self, past_key_values: Tuple, session_id: Optional[str] = None) -> str:
        """
        Create a new cache entry.
        
        Args:
            past_key_values: The KV cache to store (will be moved to CPU)
            session_id: Optional custom session ID. If None, UUID is generated.
            
        Returns:
            The session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Move KV cache to CPU to save GPU memory
        past_kv_cpu = _move_kv_to_device(past_key_values, 'cpu')
        
        now = time.time()
        with self._lock:
            self._cache[session_id] = {
                "past_key_values": past_kv_cpu,
                "created_at": now,
                "last_accessed": now,
            }
        
        return session_id
    
    def get(self, session_id: str, device: Union[str, torch.device] = 'cuda:0') -> Optional[Tuple]:
        """
        Retrieve KV cache by session ID and migrate to target device.
        
        Updates last_accessed timestamp on access.
        
        Args:
            session_id: Session identifier
            device: Target device to move KV cache to (default: 'cuda:0')
        
        Returns:
            The past_key_values tuple moved to target device, or None if not found/expired
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
            past_kv_cpu = entry["past_key_values"]
        
        # Migrate from CPU to GPU (outside lock to avoid blocking other operations)
        past_kv_gpu = _move_kv_to_device(past_kv_cpu, device)
        
        # Convert back to DynamicCache for modern transformers models (e.g., Qwen3)
        # that expect Cache objects with get_seq_length() method
        if DynamicCache is not None and isinstance(past_kv_gpu, tuple):
            past_kv_gpu = DynamicCache.from_legacy_cache(past_kv_gpu)
        
        return past_kv_gpu
    
    def update(self, session_id: str, past_key_values: Tuple) -> bool:
        """
        Update existing cache entry.
        
        Args:
            past_key_values: The KV cache to store (will be moved to CPU)
        
        Returns:
            True if updated, False if session not found
        """
        # Move KV cache to CPU to save GPU memory
        past_kv_cpu = _move_kv_to_device(past_key_values, 'cpu')
        
        with self._lock:
            if session_id not in self._cache:
                return False
            
            self._cache[session_id]["past_key_values"] = past_kv_cpu
            self._cache[session_id]["last_accessed"] = time.time()
            return True
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a cache entry.
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._cache:
                del self._cache[session_id]
                return True
            return False
    
    def exists(self, session_id: str) -> bool:
        """Check if session exists and is not expired."""
        return self.get(session_id) is not None
    
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


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(ttl_seconds: int = 1800) -> CacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(ttl_seconds=ttl_seconds)
        _cache_manager.start_cleanup_thread()
    return _cache_manager
