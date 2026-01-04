"""
KV Cache Manager for LatentMAS API

Manages server-side storage of KV caches with TTL expiration.
"""

import time
import uuid
import threading
from typing import Dict, Optional, Tuple, Any


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
            past_key_values: The KV cache to store
            session_id: Optional custom session ID. If None, UUID is generated.
            
        Returns:
            The session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        now = time.time()
        with self._lock:
            self._cache[session_id] = {
                "past_key_values": past_key_values,
                "created_at": now,
                "last_accessed": now,
            }
        
        return session_id
    
    def get(self, session_id: str) -> Optional[Tuple]:
        """
        Retrieve KV cache by session ID.
        
        Updates last_accessed timestamp on access.
        
        Returns:
            The past_key_values tuple, or None if not found/expired
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
            return entry["past_key_values"]
    
    def update(self, session_id: str, past_key_values: Tuple) -> bool:
        """
        Update existing cache entry.
        
        Returns:
            True if updated, False if session not found
        """
        with self._lock:
            if session_id not in self._cache:
                return False
            
            self._cache[session_id]["past_key_values"] = past_key_values
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
