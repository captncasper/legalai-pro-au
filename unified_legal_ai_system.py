#!/usr/bin/env python3
"""
Unified Australian Legal AI System
Combines all existing features into one powerful API
"""

import asyncio
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import your existing components
from load_real_aussie_corpus import corpus, AustralianLegalCorpus
from fix_judge_extraction import ImprovedJudgeAnalyzer
from extract_settlement_amounts import SettlementExtractor

# Import features from your existing files
try:
    # Try to import from legal_ai_mega.py
    from legal_ai_mega import (
        SemanticSearchEngine,
        QuantumLegalPredictor,
        LegalDocumentGenerator,
        SettlementCalculator
    )
    MEGA_FEATURES_AVAILABLE = True
except ImportError:
    MEGA_FEATURES_AVAILABLE = False
    print("⚠️  Could not import from legal_ai_mega.py, using fallback")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Australian Legal AI SUPREME - Unified System",
    description="All features combined: Semantic Search, Quantum Predictions, Judge Analysis, Settlement Extraction",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Request/Response Models =====
class SearchRequest(BaseModel):
    query: str
    jurisdiction: Optional[str] = "all"
    limit: Optional[int] = 20
    search_type: Optional[str] = "semantic"  # "semantic" or "keyword"

class PredictionRequest(BaseModel):
    case_description: str
    jurisdiction: Optional[str] = "nsw"
    case_type: Optional[str] = "general"
    evidence_strength: Optional[float] = 0.7

class AnalysisRequest(BaseModel):
    case_name: Optional[str] = ""
    citation: Optional[str] = ""
    description: str
    jurisdiction: Optional[str] = "nsw"

# ===== Core Services =====
class UnifiedLegalAI:
    """Unified service combining all features"""
    
    def __init__(self):
        logger.info("🚀 Initializing Unified Legal AI System...")
        
        # Load corpus
        self.corpus = corpus
        self.corpus.load_corpus()
        logger.info(f"✅ Loaded {len(self.corpus.cases)} cases")
        
        # Initialize services
        self.judge_analyzer = ImprovedJudgeAnalyzer()
        self.settlement_extractor = SettlementExtractor()
        
        # Initialize semantic search
        self._init_semantic_search()
        
        # Initialize quantum predictor
        self._init_quantum_predictor()
        
        # Cache for performance
        self.cache = {}
        
        logger.info("✅ All services initialized")
    
    def _init_semantic_search(self):
        """Initialize semantic search with fallback"""
        try:
            if MEGA_FEATURES_AVAILABLE:
                self.semantic_search = SemanticSearchEngine()
            else:
                # Fallback implementation
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.case_embeddings = None
                self._create_embeddings()
        except Exception as e:
            logger.warning(f"Semantic search init failed: {e}")
            self.embedder = None
    
    def _init_quantum_predictor(self):
        """Initialize quantum predictor with fallback"""
        try:
            if MEGA_FEATURES_AVAILABLE:
                self.quantum_predictor = QuantumLegalPredictor()
            else:
                self.quantum_predictor = None
        except Exception as e:
            logger.warning(f"Quantum predictor init failed: {e}")
            self.quantum_predictor = None
    
    def _create_embeddings(self):
        """Create embeddings for semantic search"""
        if not self.embedder:
            return
            
        embeddings_file = Path("case_embeddings.pkl")
        
        if embeddings_file.exists():
            logger.info("Loading existing embeddings...")
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                self.case_embeddings = data['embeddings']
        else:
            logger.info("Creating new embeddings...")
            texts = []
            for case in self.corpus.cases:
                text = f"{case['case_name']} {case['text']} {case['outcome']}"
                texts.append(text)
            
            self.case_embeddings = self.embedder.encode(texts, show_progress_bar=True)
            
            # Save embeddings
            with open(embeddings_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.case_embeddings,
                    'created': datetime.now().isoformat()
                }, f)
    
    async def search_cases(self, query: str, jurisdiction: str = "all", 
                          limit: int = 20, search_type: str = "semantic") -> List[Dict]:
        """Unified search with semantic and keyword options"""
        
        # Check cache
        cache_key = f"search:{query}:{jurisdiction}:{search_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = []
        
        if search_type == "semantic" and self.embedder and self.case_embeddings is not None:
            # Semantic search
            query_embedding = self.embedder.encode([query])
            similarities = np.dot(self.case_embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-limit:][::-1]
            
            for idx in top_indices:
                case = self.corpus.cases[idx]
                if jurisdiction == "all" or jurisdiction.lower() in case.get('court', '').lower():
                    # Extract settlement amounts for this case
                    amounts = self.settlement_extractor.extract_amounts(case['text'])
                    
                    results.append({
                        'case': case,
                        'similarity_score': float(similarities[idx]),
                        'settlement_amounts': amounts,
                        'max_settlement': max(amounts) if amounts else None
                    })
        else:
            # Keyword search fallback
            results_data = self.corpus.search_cases(query, 
                                                   {'court': jurisdiction} if jurisdiction != "all" else None)
            for case in results_data[:limit]:
                amounts = self.settlement_extractor.extract_amounts(case['text'])
                results.append({
                    'case': case,
                    'similarity_score': 0.8,  # Default score for keyword matches
                    'settlement_amounts': amounts,
                    'max_settlement': max(amounts) if amounts else None
                })
        
        # Cache results
        self.cache[cache_key] = results
        
        return results
    
    async def predict_outcome(self, case_description: str, jurisdiction: str = "nsw",
                            case_type: str = "general", evidence_strength: float = 0.7) -> Dict:
        """Predict case outcome with quantum analysis"""
        
        if self.quantum_predictor:
            # Use existing quantum predictor
            result = await self.quantum_predictor.predict({
                'description': case_description,
                'jurisdiction': jurisdiction,
                'case_type': case_type,
                'evidence_strength': evidence_strength
            })
            return result
        else:
            # Fallback prediction based on corpus statistics
            similar_cases = await self.search_cases(case_description, jurisdiction, limit=10)
            
            if similar_cases:
                # Calculate outcome probabilities from similar cases
                outcomes = {'applicant_won': 0, 'settled': 0, 'applicant_lost': 0}
                
                for result in similar_cases:
                    outcome = result['case'].get('outcome', 'unknown')
                    if outcome in outcomes:
                        outcomes[outcome] += result['similarity_score']
                
                # Normalize
                total = sum(outcomes.values())
                if total > 0:
                    for outcome in outcomes:
                        outcomes[outcome] /= total
                
                # Apply evidence strength modifier
                if evidence_strength > 0.7:
                    outcomes['applicant_won'] *= 1.2
                    outcomes['applicant_lost'] *= 0.8
                elif evidence_strength < 0.3:
                    outcomes['applicant_won'] *= 0.8
                    outcomes['applicant_lost'] *= 1.2
                
                # Re-normalize
                total = sum(outcomes.values())
                for outcome in outcomes:
                    outcomes[outcome] /= total
                
                return {
                    'prediction': {
                        'applicant_wins': outcomes['applicant_won'],
                        'settles': outcomes['settled'],
                        'applicant_loses': outcomes['applicant_lost'],
                        'predicted_outcome': max(outcomes, key=outcomes.get)
                    },
                    'similar_cases': [
                        {
                            'citation': r['case']['citation'],
                            'outcome': r['case']['outcome'],
                            'similarity': r['similarity_score']
                        } for r in similar_cases[:5]
                    ],
                    'confidence': max(outcomes.values()),
                    'factors': {
                        'evidence_strength': evidence_strength,
                        'jurisdiction': jurisdiction,
                        'case_type': case_type,
                        'similar_cases_found': len(similar_cases)
                    }
                }
            else:
                # No similar cases found
                return {
                    'prediction': {
                        'applicant_wins': 0.33,
                        'settles': 0.34,
                        'applicant_loses': 0.33,
                        'predicted_outcome': 'uncertain'
                    },
                    'similar_cases': [],
                    'confidence': 0.33,
                    'factors': {
                        'evidence_strength': evidence_strength,
                        'note': 'No similar cases found in corpus'
                    }
                }
    
    async def analyze_judge(self, judge_name: str) -> Dict:
        """Analyze judge patterns"""
        judge_data = self.judge_analyzer.judge_data.get(judge_name.upper(), None)
        
        if not judge_data:
            return {'error': f'No data found for Judge {judge_name}'}
        
        total_cases = judge_data['total_cases']
        
        return {
            'judge_name': judge_name.upper(),
            'total_cases': total_cases,
            'outcomes': dict(judge_data['outcomes']),
            'win_rate': judge_data['outcomes'].get('applicant_won', 0) / total_cases * 100,
            'settlement_rate': judge_data['outcomes'].get('settled', 0) / total_cases * 100,
            'case_types': dict(judge_data['case_types']),
            'recent_cases': judge_data['cases'][-5:]  # Last 5 cases
        }
    
    def get_corpus_statistics(self) -> Dict:
        """Get comprehensive corpus statistics"""
        outcome_dist = self.corpus.get_outcome_distribution()
        
        # Get settlement statistics
        settlement_data = []
        for case in self.corpus.cases[:50]:  # Sample for performance
            amounts = self.settlement_extractor.extract_amounts(case['text'])
            if amounts:
                settlement_data.append({
                    'outcome': case['outcome'],
                    'amount': max(amounts)
                })
        
        # Calculate averages by outcome
        outcome_amounts = {}
        for outcome in outcome_dist.keys():
            amounts = [s['amount'] for s in settlement_data if s['outcome'] == outcome]
            if amounts:
                outcome_amounts[outcome] = {
                    'average': sum(amounts) / len(amounts),
                    'max': max(amounts),
                    'min': min(amounts),
                    'count': len(amounts)
                }
        
        return {
            'total_cases': len(self.corpus.cases),
            'outcome_distribution': outcome_dist,
            'settlement_analysis': outcome_amounts,
            'jurisdictions': len(set(c.get('court', 'Unknown') for c in self.corpus.cases)),
            'date_range': {
                'earliest': min(c.get('year', 2000) for c in self.corpus.cases),
                'latest': max(c.get('year', 2024) for c in self.corpus.cases)
            },
            'precedent_relationships': len(self.corpus.precedent_network),
            'judges_analyzed': len(self.judge_analyzer.judge_data)
        }

# Initialize unified AI system
unified_ai = UnifiedLegalAI()

# ===== API Endpoints =====

@app.get("/")
async def root():
    """Root endpoint with system info"""
    stats = unified_ai.get_corpus_statistics()
    return {
        "system": "Australian Legal AI SUPREME - Unified",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "Semantic Search",
            "Quantum Predictions",
            "Judge Analysis",
            "Settlement Extraction",
            "Precedent Network"
        ],
        "corpus_stats": stats
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/search")
async def search_cases(request: SearchRequest):
    """Unified search endpoint"""
    try:
        results = await unified_ai.search_cases(
            query=request.query,
            jurisdiction=request.jurisdiction,
            limit=request.limit,
            search_type=request.search_type
        )
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "citation": r['case']['citation'],
                "case_name": r['case']['case_name'],
                "year": r['case']['year'],
                "court": r['case']['court'],
                "outcome": r['case']['outcome'],
                "similarity_score": r['similarity_score'],
                "snippet": r['case']['text'][:200] + "...",
                "settlement_amount": f"${r['max_settlement']:,.0f}" if r['max_settlement'] else None
            })
        
        return {
            "query": request.query,
            "search_type": request.search_type,
            "results_count": len(formatted_results),
            "results": formatted_results
        }
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict")
async def predict_outcome(request: PredictionRequest):
    """Predict case outcome"""
    try:
        result = await unified_ai.predict_outcome(
            case_description=request.case_description,
            jurisdiction=request.jurisdiction,
            case_type=request.case_type,
            evidence_strength=request.evidence_strength
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/analyze")
async def analyze_case(request: AnalysisRequest):
    """Comprehensive case analysis"""
    try:
        # Search for similar cases
        similar_cases = await unified_ai.search_cases(
            query=request.description,
            jurisdiction=request.jurisdiction,
            limit=10
        )
        
        # Get prediction
        prediction = await unified_ai.predict_outcome(
            case_description=request.description,
            jurisdiction=request.jurisdiction
        )
        
        # Extract potential settlement amounts from similar cases
        settlement_ranges = []
        for case in similar_cases:
            if case['max_settlement']:
                settlement_ranges.append(case['max_settlement'])
        
        analysis = {
            "case_info": {
                "name": request.case_name,
                "citation": request.citation,
                "jurisdiction": request.jurisdiction
            },
            "prediction": prediction['prediction'],
            "confidence": prediction['confidence'],
            "similar_cases": prediction['similar_cases'],
            "settlement_analysis": {
                "average": sum(settlement_ranges) / len(settlement_ranges) if settlement_ranges else None,
                "range": {
                    "min": min(settlement_ranges) if settlement_ranges else None,
                    "max": max(settlement_ranges) if settlement_ranges else None
                },
                "based_on": len(settlement_ranges)
            },
            "recommendations": [
                f"Review similar case: {prediction['similar_cases'][0]['citation']}" if prediction['similar_cases'] else "Gather more evidence",
                f"Expected outcome: {prediction['prediction']['predicted_outcome']}",
                f"Settlement range: ${min(settlement_ranges):,.0f} - ${max(settlement_ranges):,.0f}" if settlement_ranges else "No settlement data available"
            ]
        }
        
        return analysis
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/judge/{judge_name}")
async def get_judge_analysis(judge_name: str):
    """Get judge analysis"""
    try:
        result = await unified_ai.analyze_judge(judge_name)
        return result
    
    except Exception as e:
        logger.error(f"Judge analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/statistics")
async def get_statistics():
    """Get corpus statistics"""
    return unified_ai.get_corpus_statistics()

# ===== WebSocket for real-time features =====
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            query = json.loads(data)
            
            if query['type'] == 'search':
                results = await unified_ai.search_cases(query['query'])
                await websocket.send_json({'type': 'search_results', 'results': results[:5]})
            
            elif query['type'] == 'predict':
                prediction = await unified_ai.predict_outcome(query['description'])
                await websocket.send_json({'type': 'prediction', 'result': prediction})
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

# ===== Admin endpoints =====
@app.post("/api/v1/admin/reload-corpus")
async def reload_corpus():
    """Reload corpus data"""
    try:
        unified_ai.corpus.load_corpus()
        unified_ai._create_embeddings()
        return {"status": "success", "cases_loaded": len(unified_ai.corpus.cases)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/admin/clear-cache")
async def clear_cache():
    """Clear cache"""
    unified_ai.cache.clear()
    return {"status": "success", "message": "Cache cleared"}

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Unified Australian Legal AI System...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
