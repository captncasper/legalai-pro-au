import numpy as np
# Save as test_queries.py
import requests
import json

queries = [
    "unfair dismissal Fair Work Act",
    "directors duties Corporations Act",
    "native title Mabo",
    "contract law consideration",
    "privacy act data breach"
]

for query in queries:
    response = requests.post(
        "http://localhost:8000/search",
        json={"query": query, "num_results": 2},
        headers={"Authorization": "Bearer demo_key"}
    )
    
    result = response.json()
    print(f"\n🔍 Query: '{query}'")
    print("="*50)
    for r in result['results']:
        print(f"📊 Score: {r['relevance_score']:.3f}")
        print(f"📄 {r['document_excerpt'][:200]}...\n")