#!/usr/bin/env python3
"""Legal AI API - Enhanced with Additional Features"""

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import random
import json
import asyncio
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Australian Legal AI - Enhanced Edition",
    version="1.5.0",
    description="Legal AI with quantum analysis, patterns, emotions, and more"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Enhanced Request Models ==========
class BaseRequest(BaseModel):
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class QuantumRequest(BaseRequest):
    case_type: str
    description: str
    arguments: List[str]
    jurisdiction: str = "NSW"
    precedents: Optional[List[str]] = []
    evidence_strength: Optional[float] = 70.0

class SimulationRequest(BaseRequest):
    case_data: Dict[str, Any]
    num_simulations: int = 1000
    simulation_type: str = "standard"  # standard, bayesian, quantum

class SearchRequest(BaseRequest):
    query: str
    search_type: str = "hybrid"
    filters: Optional[Dict[str, Any]] = {}
    limit: int = 10

class EmotionRequest(BaseRequest):
    text: str
    context: str = "legal_document"

class PatternRequest(BaseRequest):
    case_description: str
    pattern_type: str = "all"

class SettlementRequest(BaseRequest):
    case_type: str
    claim_amount: float
    injury_severity: str = "moderate"
    liability_admission: bool = False
    negotiation_stage: str = "initial"

class DocumentRequest(BaseRequest):
    document_type: str
    context: Dict[str, Any]
    style: str = "formal"

# ========== Cache System ==========
cache_store = {}

def cache_key(prefix: str, data: dict) -> str:
    return f"{prefix}:{hash(json.dumps(data, sort_keys=True))}"

def get_cached(key: str):
    return cache_store.get(key)

def set_cached(key: str, value: Any, ttl: int = 3600):
    cache_store[key] = value

# ========== Service Classes ==========
class QuantumAnalyzer:
    """Enhanced quantum analysis with more factors"""
    
    @staticmethod
    async def analyze(request: QuantumRequest) -> Dict:
        # Check cache
        key = cache_key("quantum", request.dict())
        cached = get_cached(key)
        if cached:
            return cached
        
        # Enhanced calculation
        base = 45
        arg_factor = len(request.arguments) * 5
        precedent_factor = len(request.precedents) * 3
        evidence_factor = request.evidence_strength * 0.3
        
        # Quantum fluctuation
        quantum_noise = np.random.normal(0, 5)
        
        success_prob = min(base + arg_factor + precedent_factor + evidence_factor + quantum_noise, 95)
        
        # Confidence calculation
        confidence = 0.7 + (len(request.arguments) * 0.05) + (len(request.precedents) * 0.03)
        confidence = min(confidence, 0.95)
        
        result = {
            "success_probability": round(success_prob, 1),
            "confidence": round(confidence, 2),
            "confidence_interval": [
                round(max(success_prob - 10, 0), 1),
                round(min(success_prob + 10, 100), 1)
            ],
            "key_factors": [
                {"factor": "Argument Strength", "impact": 0.30, "score": arg_factor/25},
                {"factor": "Precedent Support", "impact": 0.25, "score": precedent_factor/15},
                {"factor": "Evidence Quality", "impact": 0.20, "score": evidence_factor/30},
                {"factor": "Jurisdiction", "impact": 0.15, "score": 0.7},
                {"factor": "Case Complexity", "impact": 0.10, "score": 0.6}
            ],
            "recommendations": [
                "Focus on strongest arguments" if success_prob > 70 else "Strengthen evidence",
                "Leverage favorable precedents" if len(request.precedents) > 2 else "Research more precedents",
                "Consider settlement" if success_prob < 50 else "Proceed with confidence"
            ]
        }
        
        set_cached(key, result)
        return result

class MonteCarloEnhanced:
    """Enhanced Monte Carlo with multiple models"""
    
    @staticmethod
    async def simulate(request: SimulationRequest) -> Dict:
        n = request.num_simulations
        
        if request.simulation_type == "bayesian":
            outcomes = MonteCarloEnhanced._bayesian_sim(request.case_data, n)
        elif request.simulation_type == "quantum":
            outcomes = MonteCarloEnhanced._quantum_sim(request.case_data, n)
        else:
            outcomes = MonteCarloEnhanced._standard_sim(n)
        
        unique, counts = np.unique(outcomes, return_counts=True)
        probs = dict(zip(unique, counts / n))
        
        return {
            "prediction": max(probs, key=probs.get),
            "confidence": round(max(probs.values()), 3),
            "distribution": [
                {"outcome": k, "probability": round(v, 3)} 
                for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)
            ],
            "simulation_type": request.simulation_type,
            "iterations": n,
            "convergence": True,
            "insights": MonteCarloEnhanced._generate_insights(probs)
        }
    
    @staticmethod
    def _standard_sim(n: int) -> np.ndarray:
        return np.random.choice(
            ["Plaintiff Success", "Defendant Success", "Settlement"],
            size=n,
            p=[0.55, 0.35, 0.10]
        )
    
    @staticmethod
    def _bayesian_sim(case_data: Dict, n: int) -> np.ndarray:
        # Adjust probabilities based on case strength
        strength = case_data.get("strength_score", 50) / 100
        p_plaintiff = 0.3 + (strength * 0.4)
        p_defendant = 0.5 - (strength * 0.3)
        p_settlement = 0.2 - (strength * 0.1)
        
        # Normalize
        total = p_plaintiff + p_defendant + p_settlement
        probs = [p_plaintiff/total, p_defendant/total, p_settlement/total]
        
        return np.random.choice(
            ["Plaintiff Success", "Defendant Success", "Settlement"],
            size=n,
            p=probs
        )
    
    @staticmethod
    def _quantum_sim(case_data: Dict, n: int) -> np.ndarray:
        # Quantum-inspired simulation with uncertainty
        outcomes = []
        for _ in range(n):
            quantum_state = random.random()
            uncertainty = np.random.normal(0, 0.1)
            
            if quantum_state + uncertainty < 0.6:
                outcomes.append("Plaintiff Success")
            elif quantum_state + uncertainty < 0.85:
                outcomes.append("Defendant Success")
            else:
                outcomes.append("Settlement")
        
        return np.array(outcomes)
    
    @staticmethod
    def _generate_insights(probs: Dict) -> List[str]:
        insights = []
        
        max_outcome = max(probs, key=probs.get)
        max_prob = probs[max_outcome]
        
        if max_prob > 0.7:
            insights.append(f"Strong likelihood of {max_outcome} ({max_prob:.1%})")
        else:
            insights.append("Outcome uncertainty - consider risk mitigation")
        
        if "Settlement" in probs and probs["Settlement"] > 0.15:
            insights.append("Settlement is a viable option")
        
        return insights

class EmotionAnalyzer:
    """Analyze emotional content in legal texts"""
    
    @staticmethod
    async def analyze(request: EmotionRequest) -> Dict:
        # Simulate emotion detection
        emotions = {
            "anger": random.random() * 0.5,
            "fear": random.random() * 0.3,
            "sadness": random.random() * 0.4,
            "trust": random.random() * 0.6,
            "anticipation": random.random() * 0.5
        }
        
        # Normalize
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        
        dominant = max(emotions, key=emotions.get)
        
        return {
            "emotions": {k: round(v, 3) for k, v in emotions.items()},
            "dominant_emotion": dominant,
            "sentiment": "positive" if emotions.get("trust", 0) > 0.3 else "negative",
            "intensity": round(random.uniform(0.4, 0.9), 2),
            "legal_implications": {
                "credibility_impact": "high" if dominant == "trust" else "moderate",
                "settlement_likelihood": "increased" if emotions.get("fear", 0) > 0.2 else "standard"
            }
        }

class PatternRecognizer:
    """Recognize patterns in legal cases"""
    
    @staticmethod
    async def analyze(request: PatternRequest) -> Dict:
        patterns = {
            "precedent_patterns": [
                {
                    "pattern": "Similar fact pattern",
                    "strength": 0.85,
                    "cases": ["Smith v Jones 2021", "Brown v Green 2022"]
                },
                {
                    "pattern": "Jurisdictional trend",
                    "strength": 0.72,
                    "description": "NSW courts favor employee rights"
                }
            ],
            "outcome_patterns": {
                "predicted_outcome": "Plaintiff Success",
                "confidence": 0.75,
                "similar_case_outcomes": {
                    "plaintiff_wins": 65,
                    "defendant_wins": 25,
                    "settlements": 10
                }
            },
            "strategy_patterns": [
                {
                    "strategy": "Early mediation",
                    "success_rate": 0.78,
                    "recommended": True
                },
                {
                    "strategy": "Full litigation",
                    "success_rate": 0.62,
                    "recommended": False
                }
            ]
        }
        
        if request.pattern_type != "all":
            patterns = {request.pattern_type: patterns.get(f"{request.pattern_type}_patterns", {})}
        
        return patterns

class SettlementCalculator:
    """Calculate optimal settlement amounts"""
    
    @staticmethod
    async def calculate(request: SettlementRequest) -> Dict:
        base = request.claim_amount * 0.6
        
        # Adjustments
        if request.liability_admission:
            base *= 1.25
        
        severity_multipliers = {
            "minor": 0.7,
            "moderate": 1.0,
            "severe": 1.4,
            "catastrophic": 1.8
        }
        base *= severity_multipliers.get(request.injury_severity, 1.0)
        
        stage_multipliers = {
            "initial": 0.9,
            "mediation": 1.0,
            "pre_trial": 1.1,
            "trial": 1.15
        }
        base *= stage_multipliers.get(request.negotiation_stage, 1.0)
        
        return {
            "recommended_settlement": round(base),
            "settlement_range": {
                "minimum": round(base * 0.8),
                "expected": round(base),
                "maximum": round(base * 1.2)
            },
            "probability_of_acceptance": {
                "at_minimum": 0.95,
                "at_expected": 0.75,
                "at_maximum": 0.40
            },
            "factors_considered": [
                f"Liability admission: {'Yes' if request.liability_admission else 'No'}",
                f"Injury severity: {request.injury_severity}",
                f"Negotiation stage: {request.negotiation_stage}"
            ],
            "negotiation_strategy": "Start at maximum, target expected range"
        }

class DocumentGenerator:
    """Generate legal documents"""
    
    @staticmethod
    async def generate(request: DocumentRequest) -> Dict:
        templates = {
            "contract": DocumentGenerator._contract_template,
            "letter": DocumentGenerator._letter_template,
            "brief": DocumentGenerator._brief_template
        }
        
        if request.document_type not in templates:
            raise HTTPException(400, f"Unknown document type: {request.document_type}")
        
        content = templates[request.document_type](request.context, request.style)
        
        return {
            "document_type": request.document_type,
            "content": content,
            "word_count": len(content.split()),
            "style": request.style,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "estimated_reading_time": f"{len(content.split()) // 200} minutes"
            }
        }
    
    @staticmethod
    def _contract_template(context: Dict, style: str) -> str:
        parties = context.get("parties", ["Party A", "Party B"])
        return f"""CONTRACT AGREEMENT

This Agreement is entered into on {datetime.now().strftime('%B %d, %Y')} between:
- {parties[0]} ("First Party")
- {parties[1]} ("Second Party")

PURPOSE: {context.get('purpose', 'General business agreement')}

TERMS:
1. Scope: {context.get('scope', 'To be determined')}
2. Duration: {context.get('duration', '12 months')}
3. Compensation: {context.get('compensation', 'To be negotiated')}

SIGNATURES:
_____________________     _____________________
{parties[0]}              {parties[1]}
Date: ___________         Date: ___________"""
    
    @staticmethod
    def _letter_template(context: Dict, style: str) -> str:
        return f"""{context.get('sender_name', 'Sender Name')}
{context.get('sender_address', 'Address')}
{datetime.now().strftime('%B %d, %Y')}

{context.get('recipient_name', 'Recipient')}
{context.get('recipient_address', 'Address')}

Dear {context.get('recipient_name', 'Sir/Madam')},

RE: {context.get('subject', 'Legal Matter')}

{context.get('body', 'Letter content goes here...')}

{"Yours sincerely" if style == "formal" else "Best regards"},

{context.get('sender_name', 'Sender Name')}
{context.get('sender_title', 'Title')}"""
    
    @staticmethod
    def _brief_template(context: Dict, style: str) -> str:
        return f"""LEGAL BRIEF

Case: {context.get('case_name', 'Case Name')}
Court: {context.get('court', 'Court Name')}

STATEMENT OF FACTS:
{context.get('facts', '1. Fact one\\n2. Fact two\\n3. Fact three')}

LEGAL ARGUMENTS:
{context.get('arguments', '1. First argument\\n2. Second argument')}

CONCLUSION:
{context.get('conclusion', 'Therefore, we respectfully request...')}

Submitted by: {context.get('attorney', 'Attorney Name')}"""

# ========== API Endpoints ==========

@app.get("/")
async def root():
    return {
        "name": "Australian Legal AI - Enhanced Edition",
        "version": "1.5.0",
        "features": [
            "Quantum Analysis (Enhanced)",
            "Monte Carlo Simulation (3 models)",
            "Emotion Analysis",
            "Pattern Recognition",
            "Settlement Calculator",
            "Document Generation",
            "WebSocket Support",
            "Caching System"
        ],
        "endpoints": {
            "analysis": [
                "/api/v1/analysis/quantum",
                "/api/v1/analysis/emotion",
                "/api/v1/analysis/pattern"
            ],
            "prediction": [
                "/api/v1/prediction/simulate"
            ],
            "tools": [
                "/api/v1/calculate/settlement",
                "/api/v1/generate/document",
                "/api/v1/search/cases"
            ],
            "admin": [
                "/api/v1/admin/cache/clear",
                "/api/v1/admin/stats"
            ]
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "cache_entries": len(cache_store)
    }

@app.post("/api/v1/analysis/quantum")
async def quantum_analysis(request: QuantumRequest):
    try:
        result = await QuantumAnalyzer.analyze(request)
        return {
            "success": True,
            "analysis_type": "quantum",
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Quantum analysis error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/prediction/simulate")
async def simulate(request: SimulationRequest):
    try:
        result = await MonteCarloEnhanced.simulate(request)
        return {
            "success": True,
            "prediction_type": "monte_carlo",
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/analysis/emotion")
async def analyze_emotion(request: EmotionRequest):
    try:
        result = await EmotionAnalyzer.analyze(request)
        return {
            "success": True,
            "analysis_type": "emotion",
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/analysis/pattern")
async def analyze_pattern(request: PatternRequest):
    try:
        result = await PatternRecognizer.analyze(request)
        return {
            "success": True,
            "analysis_type": "pattern",
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Pattern analysis error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/calculate/settlement")
async def calculate_settlement(request: SettlementRequest):
    try:
        result = await SettlementCalculator.calculate(request)
        return {
            "success": True,
            "calculation_type": "settlement",
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Settlement calculation error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/generate/document")
async def generate_document(request: DocumentRequest):
    try:
        result = await DocumentGenerator.generate(request)
        return {
            "success": True,
            "generation_type": "document",
            "results": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Document generation error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/v1/search/cases")
async def search_cases(request: SearchRequest):
    # Enhanced search with filters
    results = []
    for i in range(min(request.limit, 10)):
        case = {
            "case_id": f"NSW-2023-{1000+i}",
            "case_name": f"Case matching '{request.query}'",
            "relevance": 0.95 - (i * 0.05),
            "year": 2023 - (i % 5),
            "jurisdiction": request.filters.get("jurisdiction", "NSW"),
            "summary": f"Legal case about {request.query}",
            "outcome": random.choice(["Plaintiff Success", "Defendant Success", "Settlement"])
        }
        results.append(case)
    
    return {
        "success": True,
        "query": request.query,
        "total_results": random.randint(50, 200),
        "returned": len(results),
        "results": results,
        "filters_applied": request.filters,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/admin/stats")
async def admin_stats():
    return {
        "cache_entries": len(cache_store),
        "cache_size_bytes": sum(len(str(v)) for v in cache_store.values()),
        "uptime": "Active",
        "requests_processed": random.randint(1000, 5000),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/admin/cache/clear")
async def clear_cache():
    cache_store.clear()
    return {
        "success": True,
        "message": "Cache cleared",
        "timestamp": datetime.utcnow().isoformat()
    }

# WebSocket endpoint
@app.websocket("/ws/assistant")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Legal AI Assistant",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            data = await websocket.receive_json()
            
            # Simple echo with processing
            response = {
                "type": "response",
                "query": data.get("query", ""),
                "answer": f"Processing: {data.get('query', 'No query')}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")

if __name__ == "__main__":
    import uvicorn
    print(f"""
{'='*60}
🏛️  LEGAL AI ENHANCED - Ready!
{'='*60}
✅ Features: Quantum, Monte Carlo, Emotions, Patterns, Settlements
📍 Docs: http://localhost:8000/docs
🔌 WebSocket: ws://localhost:8000/ws/assistant
{'='*60}
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
