import numpy as np
#!/usr/bin/env python3
"""Performance and load testing for MEGA API"""

import asyncio
import aiohttp
import time
import statistics
from datetime import datetime

BASE_URL = "http://localhost:8000"

async def test_endpoint_performance(session, endpoint, data, num_requests=10):
    """Test endpoint performance"""
    times = []
    errors = 0
    
    for i in range(num_requests):
        start = time.time()
        try:
            async with session.post(f"{BASE_URL}{endpoint}", json=data) as response:
                await response.json()
                if response.status == 200:
                    times.append(time.time() - start)
                else:
                    errors += 1
        except Exception as e:
            errors += 1
    
    if times:
        return {
            "endpoint": endpoint,
            "requests": num_requests,
            "successful": len(times),
            "errors": errors,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "median_time": statistics.median(times)
        }
    else:
        return {"endpoint": endpoint, "errors": errors, "status": "failed"}

async def concurrent_load_test(endpoint, data, concurrent_users=5, requests_per_user=10):
    """Test with concurrent users"""
    print(f"\n🔥 Load Testing: {endpoint}")
    print(f"   Concurrent users: {concurrent_users}")
    print(f"   Requests per user: {requests_per_user}")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for user in range(concurrent_users):
            task = test_endpoint_performance(session, endpoint, data, requests_per_user)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Aggregate results
        total_requests = sum(r.get('successful', 0) for r in results)
        total_errors = sum(r.get('errors', 0) for r in results)
        
        if total_requests > 0:
            avg_response_time = statistics.mean([r['avg_time'] for r in results if 'avg_time' in r])
            print(f"   ✅ Total requests: {total_requests}")
            print(f"   ❌ Total errors: {total_errors}")
            print(f"   ⏱️  Average response time: {avg_response_time*1000:.0f}ms")
            print(f"   📊 Requests per second: {total_requests/total_time:.1f}")
        else:
            print(f"   ❌ All requests failed!")

async def run_performance_tests():
    """Run comprehensive performance tests"""
    print("🚀 MEGA Legal AI API - Performance Testing")
    print("=" * 60)
    
    # Test data
    quantum_data = {
        "case_type": "employment",
        "description": "Test case",
        "jurisdiction": "NSW",
        "arguments": ["Arg1", "Arg2", "Arg3"]
    }
    
    simulation_data = {
        "case_data": {"type": "test"},
        "num_simulations": 1000
    }
    
    # Test different endpoints under load
    await concurrent_load_test("/api/v1/analysis/quantum", quantum_data, 5, 10)
    await concurrent_load_test("/api/v1/prediction/simulate", simulation_data, 3, 5)
    await concurrent_load_test("/api/v1/search/cases", {"query": "test", "limit": 10}, 10, 20)
    
    print("\n✅ Performance testing completed!")

if __name__ == "__main__":
    asyncio.run(run_performance_tests())
