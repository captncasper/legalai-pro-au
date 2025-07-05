import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
from collections import Counter
import uvicorn

app = FastAPI(title="Australian Legal Q&A API - Enhanced")

# Load search index
with open('data/simple_index.pkl', 'rb') as f:
    search_data = pickle.load(f)
print(f"Loaded {len(search_data['documents'])} documents")

class QuestionRequest(BaseModel):
    question: str

class SearchRequest(BaseModel):
    query: str
    num_results: int = 5

# Legal knowledge templates
UNFAIR_DISMISSAL_INFO = """
**Requirements for Unfair Dismissal Claims in Australia:**

1. **Eligibility Requirements:**
   - Minimum employment period: 6 months (12 months for small business with <15 employees)
   - Must be an employee (not contractor)
   - Annual earnings below high income threshold ($175,000 as of 2024)

2. **The dismissal must be:**
   - Harsh (unreasonable in consequences)
   - Unjust (not a valid reason)
   - Unreasonable (disproportionate response)

3. **Time Limit:**
   - Must lodge within 21 days of dismissal

4. **Valid Reasons for Dismissal:**
   - Serious misconduct
   - Poor performance (with warnings)
   - Redundancy (if genuine)

"""

def search(query, num_results=5):
    words = re.findall(r'\w+', query.lower())
    doc_scores = Counter()
    
    for word in words:
        if word in search_data['keyword_index']:
            for doc_id in search_data['keyword_index'][word]:
                doc_scores[doc_id] += 1
    
    results = []
    for doc_id, score in doc_scores.most_common(num_results):
        doc = search_data['documents'][doc_id]
        
        # Extract relevant snippet around keywords
        text = doc['text']
        snippet = text[:500]
        
        # Try to find more relevant part
        for word in words:
            pos = text.lower().find(word)
            if pos > 0:
                start = max(0, pos - 100)
                end = min(len(text), pos + 400)
                snippet = "..." + text[start:end] + "..."
                break
        
        results.append({
            'text': doc['text'],
            'snippet': snippet,
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
    question = request.question.lower()
    
    # Check for specific topics with good templates
    if "requirements" in question and "unfair dismissal" in question:
        # Use template + search results
        results = search(request.question, 3)
        
        answer = UNFAIR_DISMISSAL_INFO
        
        if results:
            answer += "\n**Relevant Cases:**\n"
            for i, r in enumerate(results[:2], 1):
                # Extract case name and year
                citation = r['citation']
                answer += f"\n{i}. {citation}"
                
                # Add relevant snippet if it contains key info
                if any(word in r['snippet'].lower() for word in ['dismiss', 'unfair', 'termination']):
                    snippet = r['snippet'].replace('...', '').strip()
                    # Get first sentence about dismissal
                    sentences = snippet.split('.')
                    relevant = next((s for s in sentences if 'dismiss' in s.lower()), '')
                    if relevant:
                        answer += f"\n   - {relevant.strip()}."
        
        return {
            "answer": answer,
            "confidence": "high",
            "sources": [r['citation'] for r in results[:3]] if results else ["Fair Work Act 2009"],
            "method": "template_plus_search"
        }
    
    # General search-based answer
    results = search(request.question, 5)
    
    if results:
        answer = "Based on Australian legal documents:\n\n"
        
        # Group by relevance
        high_relevance = [r for r in results if r['score'] > 2]
        med_relevance = [r for r in results if r['score'] == 2]
        
        if high_relevance:
            answer += "**Most Relevant:**\n"
            for r in high_relevance[:2]:
                answer += f"- {r['citation']}: {r['snippet'][:200]}...\n\n"
        
        elif med_relevance:
            for r in med_relevance[:2]:
                answer += f"- {r['citation']}: {r['snippet'][:200]}...\n\n"
        
        return {
            "answer": answer,
            "confidence": "high" if high_relevance else "medium",
            "sources": [r['citation'] for r in results[:3]],
            "method": "search"
        }
    
    return {
        "answer": "I couldn't find specific information about that in the legal database. Please try rephrasing your question or contact a legal professional.",
        "confidence": "low",
        "sources": [],
        "method": "no_results"
    }

@app.get("/")
async def root():
    return {
        "message": "Australian Legal Q&A API - Enhanced",
        "endpoints": {
            "/search": "Search legal documents",
            "/ask": "Ask a legal question (with smart templates)",
            "/docs": "API documentation"
        },
        "sample_questions": [
            "What are the requirements for unfair dismissal?",
            "How long do I have to file an unfair dismissal claim?",
            "What constitutes negligence in Australian law?"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
