import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
from collections import Counter
import uvicorn

app = FastAPI(title="Australian Legal Q&A API")

# Load search index
try:
    with open('data/simple_index.pkl', 'rb') as f:
        search_data = pickle.load(f)
    print(f"Loaded {len(search_data['documents'])} documents")
except:
    search_data = None

class QuestionRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str
    num_results: int = 5

def search(query, num_results=5):
    if not search_data:
        return []
    
    words = re.findall(r'\w+', query.lower())
    doc_scores = Counter()
    
    for word in words:
        if word in search_data['keyword_index']:
            for doc_id in search_data['keyword_index'][word]:
                doc_scores[doc_id] += 1
    
    results = []
    for doc_id, score in doc_scores.most_common(num_results):
        doc = search_data['documents'][doc_id]
        results.append({
            'text': doc['text'][:500] + '...',
            'score': score,
            'citation': doc.get('metadata', {}).get('citation', 'Unknown')
        })
    return results

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    results = search(request.query, request.num_results)
    return {
        "query": request.query,
        "results": results,
        "count": len(results)
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    results = search(request.question, 3)
    if results:
        answer = f"Based on Australian legal documents:\n\n"
        answer += f"1. {results[0]['text']}\n"
        if len(results) > 1:
            answer += f"\n2. {results[1]['text']}"
        
        return {
            "answer": answer,
            "confidence": "high" if results[0]['score'] > 2 else "medium",
            "sources": [r['citation'] for r in results],
            "method": "search"
        }
    return {
        "answer": "No relevant legal documents found for your query.",
        "confidence": "low",
        "sources": [],
        "method": "search"
    }

@app.get("/")
async def root():
    return {
        "message": "Australian Legal Q&A API",
        "endpoints": {
            "/search": "Search legal documents",
            "/ask": "Ask a legal question",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
