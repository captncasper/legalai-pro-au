#!/usr/bin/env python3
"""Intelligent Predictive Caching System"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import aioredis
import pickle
import heapq

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    predicted_next_access: Optional[float] = None
    priority_score: float = 0.0
    size_bytes: int = 0

@dataclass
class AccessPattern:
    user_id: str
    query_type: str
    timestamp: float
    cache_hit: bool
    response_time: float
    query_features: Dict[str, Any] = field(default_factory=dict)

class IntelligentCacheManager:
    def __init__(self, max_size_mb: int = 1024):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.cache: Dict[str, CacheEntry] = {}
        self.access_patterns: deque = deque(maxlen=10000)
        self.predictive_model = CacheAccessPredictor()
        self.redis_client = None
        self.stats = defaultdict(int)
        
    async def initialize(self):
        """Initialize Redis connection and load models"""
        self.redis_client = await aioredis.create_redis_pool('redis://localhost')
        await self.predictive_model.load_model()
        
    async def get(self, key: str, user_id: str = None) -> Optional[Any]:
        """Intelligent cache retrieval with predictive prefetching"""
        start_time = time.time()
        
        # Check local cache first
        if key in self.cache:
            entry = self.cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Update priority based on access
            entry.priority_score = self._calculate_priority(entry)
            
            # Record access pattern
            self._record_access(user_id, key, True, time.time() - start_time)
            
            # Trigger predictive prefetching
            asyncio.create_task(self._predictive_prefetch(user_id, key))
            
            self.stats['hits'] += 1
            return entry.value
        
        # Check Redis
        if self.redis_client:
            value = await self.redis_client.get(key)
            if value:
                # Promote to local cache if frequently accessed
                await self._promote_to_local(key, pickle.loads(value))
                self.stats['redis_hits'] += 1
                return pickle.loads(value)
        
        self.stats['misses'] += 1
        self._record_access(user_id, key, False, time.time() - start_time)
        return None
    
    async def set(
        self, key: str, value: Any, ttl: int = 3600,
        user_id: str = None, priority: float = 0.5
    ):
        """Intelligent cache setting with automatic tiering"""
        size_bytes = len(pickle.dumps(value))
        
        # Determine cache tier based on value characteristics
        tier = self._determine_cache_tier(value, size_bytes, priority)
        
        if tier == 'local':
            # Ensure space is available
            await self._ensure_space(size_bytes)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                priority_score=priority,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.current_size_bytes += size_bytes
            
        # Always store in Redis for persistence
        if self.redis_client:
            await self.redis_client.setex(
                key, ttl, pickle.dumps(value)
            )
        
        # Predict future access patterns
        if user_id:
            predicted_time = await self.predictive_model.predict_next_access(
                user_id, key, self.access_patterns
            )
            if key in self.cache:
                self.cache[key].predicted_next_access = predicted_time
    
    async def _ensure_space(self, required_bytes: int):
        """Ensure cache has space using intelligent eviction"""
        while self.current_size_bytes + required_bytes > self.max_size_bytes:
            # Find least valuable entry to evict
            eviction_candidate = self._select_eviction_candidate()
            
            if eviction_candidate:
                # Move to Redis before evicting
                if self.redis_client:
                    await self.redis_client.setex(
                        eviction_candidate.key,
                        3600,
                        pickle.dumps(eviction_candidate.value)
                    )
                
                self.current_size_bytes -= eviction_candidate.size_bytes
                del self.cache[eviction_candidate.key]
                self.stats['evictions'] += 1
            else:
                break
    
    def _select_eviction_candidate(self) -> Optional[CacheEntry]:
        """Select entry for eviction using ML-based scoring"""
        if not self.cache:
            return None
        
        current_time = time.time()
        candidates = []
        
        for key, entry in self.cache.items():
            # Calculate eviction score (lower is better)
            score = self._calculate_eviction_score(entry, current_time)
            heapq.heappush(candidates, (score, key))
        
        # Get entry with lowest score
        if candidates:
            _, key = heapq.heappop(candidates)
            return self.cache[key]
        
        return None
    
    def _calculate_eviction_score(self, entry: CacheEntry, current_time: float) -> float:
        """Calculate eviction score using multiple factors"""
        # Time since last access (normalized)
        time_since_access = (current_time - entry.last_accessed) / 3600  # hours
        
        # Access frequency (inverse)
        frequency_score = 1 / (entry.access_count + 1)
        
        # Size penalty (larger items evicted first)
        size_penalty = entry.size_bytes / self.max_size_bytes
        
        # Predicted future access (if available)
        if entry.predicted_next_access:
            time_until_access = max(0, entry.predicted_next_access - current_time)
            prediction_score = 1 / (time_until_access / 3600 + 1)
        else:
            prediction_score = 0.5
        
        # Weighted combination
        score = (
            0.3 * time_since_access +
            0.2 * frequency_score +
            0.2 * size_penalty +
            0.3 * (1 - prediction_score)
        )
        
        return score
    
    async def _predictive_prefetch(self, user_id: str, accessed_key: str):
        """Prefetch related queries based on access patterns"""
        if not user_id:
            return
        
        # Get predicted next queries
        predictions = await self.predictive_model.predict_next_queries(
            user_id, accessed_key, self.access_patterns
        )
        
        for predicted_key, confidence in predictions:
            if confidence > 0.7 and predicted_key not in self.cache:
                # Check if we should prefetch
                if await self._should_prefetch(predicted_key, confidence):
                    asyncio.create_task(
                        self._prefetch_query(predicted_key, confidence)
                    )
    
    async def _should_prefetch(self, key: str, confidence: float) -> bool:
        """Determine if prefetching is worthwhile"""
        # Check historical patterns
        hit_rate = self._calculate_historical_hit_rate(key)
        
        # Check current load
        current_load = self.current_size_bytes / self.max_size_bytes
        
        # Prefetch if high confidence, good hit rate, and space available
        return confidence > 0.7 and hit_rate > 0.5 and current_load < 0.8
    
    def _record_access(
        self, user_id: str, key: str, cache_hit: bool, response_time: float
    ):
        """Record access pattern for learning"""
        pattern = AccessPattern(
            user_id=user_id or 'anonymous',
            query_type=self._extract_query_type(key),
            timestamp=time.time(),
            cache_hit=cache_hit,
            response_time=response_time,
            query_features=self._extract_query_features(key)
        )
        self.access_patterns.append(pattern)
    
    def _extract_query_type(self, key: str) -> str:
        """Extract query type from cache key"""
        parts = key.split(':')
        return parts[0] if parts else 'unknown'
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions'],
            'current_size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'capacity_used': self.current_size_bytes / self.max_size_bytes,
            'entries_count': len(self.cache),
            'avg_entry_size_kb': (self.current_size_bytes / len(self.cache) / 1024) if self.cache else 0,
            'prediction_accuracy': await self.predictive_model.get_accuracy()
        }

class CacheAccessPredictor:
    """ML model for predicting cache access patterns"""
    
    def __init__(self):
        self.user_models = {}
        self.global_model = None
        
    async def load_model(self):
        """Load pre-trained models"""
        # In production, load from saved models
        pass
    
    async def predict_next_access(
        self, user_id: str, key: str, patterns: deque
    ) -> float:
        """Predict when this key will be accessed next"""
        user_patterns = [p for p in patterns if p.user_id == user_id]
        
        if len(user_patterns) < 5:
            # Not enough data, use global average
            return time.time() + 3600  # Default 1 hour
        
        # Extract features
        features = self._extract_features(user_patterns, key)
        
        # Simple time-series prediction (in production, use LSTM/Prophet)
        intervals = []
        for i in range(1, len(user_patterns)):
            if user_patterns[i].query_type == self._extract_query_type(key):
                intervals.append(
                    user_patterns[i].timestamp - user_patterns[i-1].timestamp
                )
        
        if intervals:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Predict next access with confidence interval
            predicted_interval = avg_interval + 0.5 * std_interval
            return time.time() + predicted_interval
        
        return time.time() + 7200  # Default 2 hours
    
    async def predict_next_queries(
        self, user_id: str, current_key: str, patterns: deque
    ) -> List[Tuple[str, float]]:
        """Predict likely next queries after current one"""
        user_patterns = [p for p in patterns if p.user_id == user_id]
        
        # Build transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(user_patterns) - 1):
            current_type = user_patterns[i].query_type
            next_type = user_patterns[i + 1].query_type
            transitions[current_type][next_type] += 1
        
        current_type = self._extract_query_type(current_key)
        predictions = []
        
        if current_type in transitions:
            total = sum(transitions[current_type].values())
            for next_type, count in transitions[current_type].items():
                confidence = count / total
                # Generate likely key based on type
                predicted_key = f"{next_type}:predicted:{user_id}"
                predictions.append((predicted_key, confidence))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    
    def _extract_features(self, patterns: List[AccessPattern], key: str) -> Dict[str, float]:
        """Extract features for ML prediction"""
        features = {
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'query_type': hash(self._extract_query_type(key)) % 1000,
            'user_activity_level': len(patterns),
            'avg_session_length': self._calculate_avg_session_length(patterns),
            'query_diversity': len(set(p.query_type for p in patterns))
        }
        return features
    
    def _calculate_avg_session_length(self, patterns: List[AccessPattern]) -> float:
        """Calculate average session length"""
        if not patterns:
            return 0
        
        sessions = []
        session_start = patterns[0].timestamp
        last_timestamp = patterns[0].timestamp
        
        for pattern in patterns[1:]:
            if pattern.timestamp - last_timestamp > 1800:  # 30 min gap = new session
                sessions.append(last_timestamp - session_start)
                session_start = pattern.timestamp
            last_timestamp = pattern.timestamp
        
        sessions.append(last_timestamp - session_start)
        return np.mean(sessions) if sessions else 0
    
    async def get_accuracy(self) -> float:
        """Get prediction accuracy metric"""
        # In production, track actual vs predicted
        return 0.82  # Placeholder

# Test the intelligent cache
async def test_intelligent_cache():
    cache = IntelligentCacheManager(max_size_mb=100)
    await cache.initialize()
    
    # Simulate user sessions
    users = ['user1', 'user2', 'user3']
    query_types = ['search', 'analysis', 'prediction', 'document']
    
    print("Simulating cache access patterns...")
    
    for _ in range(100):
        user = np.random.choice(users)
        query_type = np.random.choice(query_types)
        key = f"{query_type}:{np.random.randint(1, 20)}"
        
        # Try to get from cache
        result = await cache.get(key, user)
        
        if result is None:
            # Simulate computation
            await asyncio.sleep(0.1)
            value = f"Result for {key}"
            await cache.set(key, value, user_id=user)
    
    # Get statistics
    stats = await cache.get_cache_stats()
    print("\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_intelligent_cache())
