#!/usr/bin/env python3
import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Australian Legal AI",
    description="AI-powered Australian legal research and case analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    jurisdiction: Optional[str] = "all"
    limit: Optional[int] = 20
    api_key: Optional[str] = "demo_key"

DEMO_CORPUS = [
    {
        "citation": "[2023] HCA 15",
        "case_name": "Employment Rights Case v Commonwealth",
        "court": "HCA",
        "jurisdiction": "federal",
        "summary": "Employment termination case involving unfair dismissal claims.",
        "outcome": "Appellant successful",
        "settlement_amount": 150000,
        "catchwords": "employment law, unfair dismissal, compensation"
    },
    {
        "citation": "[2023] NSWCA 89", 
        "case_name": "Property Development Dispute v Council",
        "court": "NSWCA",
        "jurisdiction": "nsw",
        "summary": "Property development dispute involving planning law.",
        "outcome": "Appeal dismissed",
        "settlement_amount": 75000,
        "catchwords": "property law, development, planning"
    }
]

VALID_API_KEYS = {"demo_key": {"tier": "demo"}}

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse("""
    <html><head><title>Australian Legal AI</title></head>
    <body style="font-family: Arial; margin: 40px; background: #f5f5f5;">
        <div style="max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px;">
            <h1>🏛️ Australian Legal AI API</h1>
            <p><span style="background: green; color: white; padding: 5px 15px; border-radius: 20px;">✅ LIVE</span></p>
            <h2>Quick Start</h2>
            <p><strong>Search:</strong> POST /api/v1/search</p>
            <p><strong>Docs:</strong> <a href="/docs">/docs</a></p>
            <p><strong>Health:</strong> <a href="/health">/health</a></p>
            <h2>Demo API Key: <code>demo_key</code></h2>
        </div>
    </body></html>
    """)

@app.get("/health")
async def health():
    return {"status": "healthy", "corpus_size": len(DEMO_CORPUS)}

@app.post("/api/v1/search")
async def search(request: SearchRequest):
    if request.api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    query_lower = request.query.lower()
    results = []
    
    for case in DEMO_CORPUS:
        if request.jurisdiction != "all" and case["jurisdiction"] != request.jurisdiction:
            continue
            
        searchable = f"{case['case_name']} {case['summary']} {case['catchwords']}".lower()
        score = sum(1 for word in query_lower.split() if word in searchable)
        
        if score > 0:
            case_result = case.copy()
            case_result["relevance_score"] = score
            results.append(case_result)
    
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "status": "success",
        "results": results[:request.limit],
        "total": len(results),
        "query": request.query
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
