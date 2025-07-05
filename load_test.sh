#!/bin/bash

echo "🔥 Running Load Test..."

python3 << 'PYTHON'
import asyncio
import aiohttp
import time
import random

async def simulate_user(session, user_id):
    """Simulate a user making various requests"""
    queries = [
        {"endpoint": "/api/v1/search/cases", "data": {"query": f"user_{user_id} contract"}},
        {"endpoint": "/api/v1/analysis/quantum-supreme", "data": {"case_name": f"Case_{user_id}"}},
        {"endpoint": "/api/v1/prediction/simulate", "data": {"scenario": "test"}}
    ]
    
    results = []
    for query in queries:
        start = time.time()
        try:
            async with session.post(
                f"http://localhost:8000{query['endpoint']}",
                json=query['data']
            ) as resp:
                await resp.json()
                elapsed = time.time() - start
                results.append({"user": user_id, "time": elapsed, "status": resp.status})
        except Exception as e:
            results.append({"user": user_id, "time": 0, "status": "error"})
    
    return results

async def load_test(num_users=10):
    async with aiohttp.ClientSession() as session:
        print(f"Simulating {num_users} concurrent users...")
        
        start = time.time()
        tasks = [simulate_user(session, i) for i in range(num_users)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start
        
        # Analyze results
        all_times = [r['time'] for user_results in results for r in user_results if r['time'] > 0]
        success_count = sum(1 for user_results in results for r in user_results if r['status'] == 200)
        total_requests = sum(len(user_results) for user_results in results)
        
        print(f"\n📊 Load Test Results:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Requests/sec: {total_requests/total_time:.1f}")
        print(f"  Success Rate: {success_count/total_requests:.1%}")
        print(f"  Avg Response: {sum(all_times)/len(all_times):.3f}s")
        print(f"  Max Response: {max(all_times):.3f}s")

asyncio.run(load_test(20))
PYTHON
