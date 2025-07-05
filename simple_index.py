import numpy as np
import json
import re
from collections import Counter
import pickle
import os

print("Building simple legal search index...")
os.makedirs('data', exist_ok=True)

documents = []
keyword_index = {}

with open('corpus_export/australian_legal_corpus.jsonl', 'r') as f:
    for i, line in enumerate(f):
        doc = json.loads(line.strip())
        doc_id = len(documents)
        documents.append(doc)
        
        # Extract keywords
        text = doc['text'].lower()
        words = re.findall(r'\w+', text)
        
        # Index each unique word
        for word in set(words):
            if len(word) > 2:  # Skip short words
                if word not in keyword_index:
                    keyword_index[word] = []
                keyword_index[word].append(doc_id)
        
        if i % 1000 == 0:
            print(f"Indexed {i} documents...")

# Save index
with open('data/simple_index.pkl', 'wb') as f:
    pickle.dump({
        'documents': documents,
        'keyword_index': keyword_index
    }, f)

print(f"Success! Indexed {len(documents)} documents")
print(f"Index size: {len(keyword_index)} unique terms")

# Test search function
def search(query, num_results=5):
    query_words = re.findall(r'\w+', query.lower())
    doc_scores = Counter()
    
    for word in query_words:
        if word in keyword_index:
            for doc_id in keyword_index[word]:
                doc_scores[doc_id] += 1
    
    results = []
    for doc_id, score in doc_scores.most_common(num_results):
        doc = documents[doc_id]
        results.append({
            'text': doc['text'][:200] + '...',
            'metadata': doc.get('metadata', {}),
            'score': score
        })
    return results

# Test it
print("\nTesting search for 'unfair dismissal':")
for r in search('unfair dismissal'):
    print(f"Score: {r['score']} - {r['text'][:100]}...")
