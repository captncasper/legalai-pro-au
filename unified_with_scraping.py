#!/usr/bin/env python3
"""
Unified Australian Legal AI System with Intelligent Scraping
This version includes all the scraping features
"""

# Copy the entire content from unified_legal_ai_system_fixed.py first
# Then add the scraping features

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

# Import your working components
from load_real_aussie_corpus import corpus, AustralianLegalCorpus
from extract_settlement_amounts import SettlementExtractor

# Import scraping components
from intelligent_legal_scraper import IntelligentLegalScraper, ScrapingIntegration

# Import sentence transformers for semantic search
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("⚠️  sentence-transformers not available, using keyword search only")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Australian Legal AI SUPREME - Unified System with Scraping",
    description="All features: Semantic Search, Predictions, Judge Analysis, Settlement Extraction, Intelligent Scraping",
    version="2.0.0"
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

# ===== Fixed Judge Analyzer =====
class ImprovedJudgeAnalyzer:
    def __init__(self):
        self.judge_data = {}
        self.analyzed = False
        
    def extract_judge_name(self, text):
        """Better extraction of judge names"""
        import re
        
        # More specific patterns for Australian courts
        patterns = [
            # "Coram: Smith J" or "Coram: Smith CJ"
            r'Coram:\s*([A-Z][a-z]+)\s+(?:CJ|J|JA|P|DP|JJ)\b',
            # "Before: Justice Smith" or "Before: Judge Smith"
            r'Before:\s*(?:Justice|Judge|The Hon(?:ourable)?\.?)\s+([A-Z][a-z]+)',
            # "Smith J:" at start of line
            r'^([A-Z][a-z]+)\s+(?:CJ|J|JA|P|DP|JJ):\s',
            # "The Honourable Justice Smith"
            r'The\s+Hon(?:ourable)?\.?\s+(?:Justice|Judge)\s+([A-Z][a-z]+)',
            # "SMITH J" in all caps
            r'\b([A-Z]{3,})\s+J\b(?!\w)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                # Return the first valid match
                for match in matches:
                    # Filter out common false positives
                    if match.upper() not in ['THE', 'OF', 'IN', 'TO', 'AND', 'OR', 'FOR', 'BY', 
                                           'WITH', 'FROM', 'BORDER', 'APPLICANT', 'RESPONDENT',
                                           'AUSTRALIA', 'GROUP', 'SERVICES', 'PTY', 'LTD', 'LIMITED']:
                        return match.upper()
        
        return None
    
    def analyze_all_judges(self, cases):
        """Analyze all judges in the corpus"""
        from collections import defaultdict
        
        self.judge_data = defaultdict(lambda: {
            'cases': [],
            'outcomes': defaultdict(int),
            'case_types': defaultdict(int),
            'total_cases': 0
        })
        
        for case in cases:
            judge_name = self.extract_judge_name(case['text'])
            
            if judge_name:
                self.judge_data[judge_name]['cases'].append(case['citation'])
                self.judge_data[judge_name]['outcomes'][case['outcome']] += 1
                self.judge_data[judge_name]['total_cases'] += 1
                
                # Determine case type
                case_type = self._determine_case_type(case['text'])
                self.judge_data[judge_name]['case_types'][case_type] += 1
        
        self.analyzed = True
        return dict(self.judge_data)
    
    def _determine_case_type(self, text):
        """Determine case type from text"""
        text_lower = text.lower()
        
        case_types = {
            'negligence': ['negligence', 'injury', 'accident'],
            'contract': ['contract', 'breach', 'agreement'],
            'employment': ['employment', 'dismissal', 'workplace'],
            'property': ['property', 'land', 'real estate'],
            'immigration': ['immigration', 'visa', 'refugee'],
            'criminal': ['criminal', 'offence', 'prosecution']
        }
        
        for case_type, keywords in case_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return case_type
        
        return 'other'

# ===== Core Services =====
class UnifiedLegalAI:
    """Unified service combining all features including scraping"""
    
    def __init__(self):
        logger.info("🚀 Initializing Unified Legal AI System with Scraping...")
        
        # Load corpus
        self.corpus = corpus
        self.corpus.load_corpus()
        logger.info(f"✅ Loaded {len(self.corpus.cases)} cases")
        
        # Initialize services
        self.judge_analyzer = ImprovedJudgeAnalyzer()
        self.settlement_extractor = SettlementExtractor()
        
        # Initialize semantic search if available
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.case_embeddings = None
            self._create_embeddings()
        else:
            self.embedder = None
            self.case_embeddings = None
        
        # Analyze judges
        self.judge_analyzer.analyze_all_judges(self.corpus.cases)
        logger.info(f"✅ Analyzed {len(self.judge_analyzer.judge_data)} judges")
        
        # Initialize scraping integration
        self.scraping_integration = ScrapingIntegration(self.corpus)
        self.auto_scrape_enabled = True
        logger.info("✅ Scraping integration initialized")
        
        # Cache for performance
        self.cache = {}
        
        logger.info("✅ All services initialized")
    
    def _create_embeddings(self):
        """Create embeddings for semantic search"""
        if not self.embedder:
            return
            
        embeddings_file = Path("case_embeddings.pkl")
        
        if embeddings_file.exists():
            logger.info("Loading existing embeddings...")
            try:
                with open(embeddings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.case_embeddings = data['embeddings']
                logger.info("✅ Embeddings loaded")
            except Exception as e:
                logger.error(f"Could not load embeddings: {e}")
                self._create_new_embeddings()
        else:
            self._create_new_embeddings()
    
    def _create_new_embeddings(self):
        """Create new embeddings"""
        logger.info("Creating new embeddings...")
        texts = []
        for case in self.corpus.cases:
            text = f"{case['case_name']} {case['text']} {case['outcome']}"
            texts.append(text[:512])  # Limit length
        
        try:
            self.case_embeddings = self.embedder.encode(texts, show_progress_bar=True)
            
            # Save embeddings
            with open("case_embeddings.pkl", 'wb') as f:
                pickle.dump({
                    'embeddings': self.case_embeddings,
                    'created': datetime.now().isoformat()
                }, f)
            logger.info("✅ Embeddings created and saved")
        except Exception as e:
            logger.error(f"Could not create embeddings: {e}")
            self.case_embeddings = None
    
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
            search_results = self.corpus.search_cases(query)
            for case in search_results[:limit]:
                if jurisdiction == "all" or jurisdiction.lower() in case.get('court', '').lower():
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
        """Predict case outcome based on similar cases"""
        
        # Find similar cases
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

# ===== Original API Endpoints =====

@app.get("/")
async def root():
    """Root endpoint with system info"""
    stats = unified_ai.get_corpus_statistics()
    return {
        "system": "Australian Legal AI SUPREME - Unified with Scraping",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Semantic Search" if SENTENCE_TRANSFORMER_AVAILABLE else "Keyword Search",
            "Case Outcome Prediction",
            "Judge Analysis",
            "Settlement Extraction",
            "Precedent Network",
            "Intelligent Scraping"  # NEW!
        ],
        "corpus_stats": stats,
        "scraping_enabled": unified_ai.auto_scrape_enabled
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
            search_type=request.search_type if SENTENCE_TRANSFORMER_AVAILABLE else "keyword"
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
            "search_type": request.search_type if SENTENCE_TRANSFORMER_AVAILABLE else "keyword",
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

# ===== NEW SCRAPING ENDPOINTS =====

@app.post("/api/v1/search/smart")
async def smart_search_with_scraping(request: SearchRequest):
    """Search with automatic scraping if needed"""
    try:
        # First try normal search
        results = await unified_ai.search_cases(
            query=request.query,
            jurisdiction=request.jurisdiction,
            limit=request.limit
        )
        
        # If not enough results and auto-scrape is enabled
        if len(results) < 5 and unified_ai.auto_scrape_enabled:
            logger.info(f"Only {len(results)} results found, triggering smart scraping...")
            
            # Use scraping integration
            all_results = await unified_ai.scraping_integration.search_with_scraping(
                query=request.query,
                jurisdiction=request.jurisdiction,
                limit=request.limit
            )
            
            # Format results
            formatted_results = []
            for r in all_results:
                if isinstance(r, dict) and 'citation' in r:
                    # Scraped result
                    formatted_results.append({
                        "citation": r['citation'],
                        "case_name": r.get('case_name', r.get('title', '')),
                        "year": r.get('year', 0),
                        "court": r.get('court', ''),
                        "source": r.get('source', 'corpus'),
                        "snippet": r.get('text', '')[:200] + "...",
                        "url": r.get('url', '')
                    })
            
            return {
                "query": request.query,
                "search_type": "smart_scraping",
                "results_count": len(formatted_results),
                "scraped_new": len([r for r in formatted_results if r['source'] == 'scraped']),
                "results": formatted_results
            }
        else:
            # Normal results
            return await search_cases(request)
            
    except Exception as e:
        logger.error(f"Smart search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scrape/case")
async def scrape_specific_case(citation: str):
    """Scrape a specific case by citation"""
    try:
        async with IntelligentLegalScraper() as scraper:
            case = await scraper.fetch_specific_case(citation)
            
            if case:
                # Add to corpus
                await unified_ai.scraping_integration._add_to_corpus([case])
                
                return {
                    "status": "success",
                    "case": case
                }
            else:
                raise HTTPException(status_code=404, detail=f"Case {citation} not found")
                
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/scrape/topic")
async def scrape_topic(topic: str, max_cases: int = 20):
    """Scrape cases about a specific topic"""
    try:
        async with IntelligentLegalScraper() as scraper:
            cases = await scraper.smart_search(topic, {'max_results': max_cases})
            
            # Add to corpus
            if cases:
                await unified_ai.scraping_integration._add_to_corpus(cases)
            
            return {
                "status": "success",
                "topic": topic,
                "cases_found": len(cases),
                "cases": cases
            }
            
    except Exception as e:
        logger.error(f"Topic scraping error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/scrape/status")
async def get_scraping_status():
    """Get scraping status and statistics"""
    scraped_dir = Path("scraped_cases")
    scraped_count = 0
    
    if scraped_dir.exists():
        for file in scraped_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    scraped_count += len(data) if isinstance(data, list) else 1
            except:
                pass
    
    return {
        "auto_scrape_enabled": unified_ai.auto_scrape_enabled,
        "scraped_cases_count": scraped_count,
        "corpus_size": len(unified_ai.corpus.cases),
        "total_available": len(unified_ai.corpus.cases) + scraped_count
    }

@app.post("/api/v1/scrape/toggle")
async def toggle_auto_scraping(enabled: bool):
    """Enable or disable automatic scraping"""
    unified_ai.auto_scrape_enabled = enabled
    return {
        "status": "success",
        "auto_scrape_enabled": enabled
    }

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
        if SENTENCE_TRANSFORMER_AVAILABLE:
            unified_ai._create_embeddings()
        return {"status": "success", "cases_loaded": len(unified_ai.corpus.cases)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/admin/clear-cache")
async def clear_cache():
    """Clear cache"""
    unified_ai.cache.clear()
    return {"status": "success", "message": "Cache cleared"}

# ===== Main entry point =====
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Unified Australian Legal AI System with Scraping...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
