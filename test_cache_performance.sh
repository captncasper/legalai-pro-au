#!/bin/bash

echo "🧠 Testing Intelligent Cache Performance..."

python3 << 'PYTHON'
import asyncio
import aiohttp
import time
import statistics

async def test_cache():
    queries = [
        "contract breach NSW damages",
        "negligence compensation victoria",
        "employment unfair dismissal",
        "contract breach NSW damages",  # Repeat for cache hit
        "negligence compensation victoria"  # Repeat for cache hit
    ]
    
    response_times = []
    
    async with aiohttp.ClientSession() as session:
        for i, query in enumerate(queries):
            start = time.time()
            
            async with session.post(
                "http://localhost:8000/api/v1/search/cases",
                json={"query": query, "jurisdiction": "all"}
            ) as resp:
                await resp.json()
                elapsed = time.time() - start
                response_times.append(elapsed)
                
                cache_status = "HIT" if i >= 3 else "MISS"
                print(f"Query {i+1}: {elapsed:.3f}s ({cache_status})")
        
        # Get cache stats
        async with session.get("http://localhost:8000/api/v1/admin/stats") as resp:
            stats = await resp.json()
            if "cache_stats" in stats:
                print(f"\n📊 Cache Statistics:")
                print(f"  Hit Rate: {stats['cache_stats'].get('hit_rate', 0):.2%}")
                print(f"  Entries: {stats['cache_stats'].get('entries_count', 0)}")
    
    # Analyze performance
    first_calls = response_times[:3]
    cached_calls = response_times[3:]
    
    if cached_calls:
        speedup = statistics.mean(first_calls) / statistics.mean(cached_calls)
        print(f"\n⚡ Cache Speedup: {speedup:.1f}x faster")

asyncio.run(test_cache())
PYTHON
