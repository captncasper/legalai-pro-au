import numpy as np
#!/usr/bin/env python3
"""
ULTIMATE Legal API - Combines ALL features:
- Original search
- Smart AI predictions
- RAG with citations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import re
from collections import Counter
import uvicorn
from legal_rag import LegalRAG

app = FastAPI(
    title="Ultimate Australian Legal AI",
    description="🚀 Search + Smart AI + RAG = Complete Legal Solution",
    version="4.0"
)

# Load your original search index
with open('data/simple_index.pkl', 'rb') as f:
    search_data = pickle.load(f)
    documents = search_data['documents']

# Initialize RAG
rag_engine = LegalRAG()

# ============= MODELS =============
class SearchRequest(BaseModel):
    query: str
    n_results: int = 5

class PredictRequest(BaseModel):
    case_details: str

class RAGRequest(BaseModel):
    question: str
    n_sources: int = 5

# ============= ORIGINAL SEARCH =============
def keyword_search(query: str, n_results: int = 5) -> List[Dict]:
    """Your original keyword search"""
    words = re.findall(r'\w+', query.lower())
    doc_scores = Counter()
    
    for word in words:
        if word in search_data['keyword_index']:
            for doc_id in search_data['keyword_index'][word]:
                doc_scores[doc_id] += 1
    
    results = []
    for doc_id, score in doc_scores.most_common(n_results):
        doc = documents[doc_id]
        results.append({
            'text': doc['text'][:500] + '...',
            'score': score,
            'citation': doc.get('metadata', {}).get('citation', 'Unknown'),
            'method': 'keyword_search'
        })
    return results

# ============= SMART AI PREDICTIONS =============
def predict_outcome(case_details: str) -> Dict:
    """Smart case outcome prediction"""
    case_lower = case_details.lower()
    score = 50  # Base score
    factors = []
    
    # Positive indicators
    if 'no warning' in case_lower:
        score += 20
        factors.append("✓ No warnings given (+20%)")
    if 'long service' in case_lower or re.search(r'\d+\s*years', case_lower):
        score += 15
        factors.append("✓ Long service (+15%)")
    if 'good performance' in case_lower:
        score += 10
        factors.append("✓ Good performance history (+10%)")
    
    # Negative indicators
    if 'misconduct' in case_lower:
        score -= 30
        factors.append("✗ Misconduct alleged (-30%)")
    if 'small business' in case_lower:
        score -= 10
        factors.append("✗ Small business employer (-10%)")
    
    return {
        'success_probability': min(max(score, 5), 95),
        'factors': factors,
        'recommendation': "Strong case - proceed" if score > 70 else "Moderate case - gather evidence" if score > 40 else "Weak case - consider settlement",
        'method': 'smart_prediction'
    }

# ============= API ENDPOINTS =============

@app.get("/")
async def root():
    return {
        "message": "🚀 Ultimate Legal AI - All Features Combined!",
        "endpoints": {
            "search": {
                "/search/keyword": "Original keyword search",
                "/search/semantic": "RAG semantic search with citations"
            },
            "ai": {
                "/predict": "Predict case outcome",
                "/analyze": "Complete case analysis"
            },
            "rag": {
                "/ask": "Ask question with cited sources",
                "/chat": "Legal chat with RAG"
            }
        },
        "stats": {
            "documents": len(documents),
            "rag_chunks": rag_engine.collection.count()
        }
    }

# Original search endpoint
@app.post("/search/keyword")
async def search_keyword(request: SearchRequest):
    """Original keyword-based search"""
    return {
        "query": request.query,
        "results": keyword_search(request.query, request.n_results),
        "method": "keyword"
    }

# RAG search endpoint
@app.post("/search/semantic")
async def search_semantic(request: SearchRequest):
    """Semantic search with RAG"""
    result = rag_engine.query(request.query, request.n_results)
    return {
        "query": request.query,
        "results": result['sources'],
        "method": "semantic_rag"
    }

# Smart prediction endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    """Predict case outcome"""
    return predict_outcome(request.case_details)

# RAG Q&A endpoint
@app.post("/ask")
async def ask(request: RAGRequest):
    """Ask question and get answer with citations"""
    return rag_engine.query(request.question, request.n_sources)

# Combined analysis endpoint
@app.post("/analyze")
async def analyze(request: PredictRequest):
    """Complete analysis: prediction + search + RAG"""
    case_details = request.case_details
    
    # 1. Predict outcome
    prediction = predict_outcome(case_details)
    
    # 2. Keyword search
    keyword_results = keyword_search(case_details, 3)
    
    # 3. RAG search
    rag_result = rag_engine.query(case_details, 3)
    
    return {
        "case_details": case_details,
        "prediction": prediction,
        "keyword_matches": keyword_results,
        "semantic_sources": rag_result['sources'],
        "rag_answer": rag_result['answer'],
        "recommendations": [
            f"Success probability: {prediction['success_probability']}%",
            f"Found {len(keyword_results)} keyword matches",
            f"Found {len(rag_result['sources'])} semantic matches",
            "Consider cited cases for precedent"
        ]
    }

# Chat endpoint
@app.post("/chat")
async def chat(message: str):
    """Chat interface using RAG"""
    result = rag_engine.query(message)
    
    return {
        "user": message,
        "assistant": result['answer'],
        "sources_used": len(result['sources']),
        "confidence": "high" if result['sources'] else "low"
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 ULTIMATE LEGAL AI API")
    print("=" * 60)
    print("✅ Original keyword search")
    print("✅ Smart AI predictions")
    print("✅ RAG with real citations")
    print("✅ Everything in ONE API!")
    print("=" * 60)
    print("Starting on http://localhost:8000")
    print("API docs at http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
