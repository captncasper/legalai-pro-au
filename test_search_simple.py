import numpy as np
#!/usr/bin/env python3
import pickle
import re
from collections import Counter
import json

# Load index
with open('data/simple_index.pkl', 'rb') as f:
    data = pickle.load(f)

def search(query):
    words = re.findall(r'\w+', query.lower())
    doc_scores = Counter()
    
    for word in words:
        if word in data['keyword_index']:
            for doc_id in data['keyword_index'][word]:
                doc_scores[doc_id] += 1
    
    results = []
    for doc_id, score in doc_scores.most_common(3):
        doc = data['documents'][doc_id]
        results.append({
            'text': doc['text'][:300] + '...',
            'score': score,
            'citation': doc.get('metadata', {}).get('citation', 'Unknown')
        })
    
    return results

# Test queries
queries = [
    "unfair dismissal",
    "employment contract", 
    "negligence"
]

for q in queries:
    print(f"\n🔍 Query: {q}")
    results = search(q)
    if results:
        print(f"✓ Top result (score: {results[0]['score']}):")
        print(f"  {results[0]['text']}")
        print(f"  Citation: {results[0]['citation']}")
    else:
        print("✗ No results")
