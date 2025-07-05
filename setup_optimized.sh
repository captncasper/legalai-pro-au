#!/bin/bash
# Simple setup script for Optimized Legal AI API

echo "🏗️ Setting up Optimized Legal AI API..."

# Create directories
mkdir -p app/{routers,core,services,models,utils}

# Create the main optimized API file
cat > optimized_main.py << 'MAINEOF'
#!/usr/bin/env python3
"""Australian Legal AI API - Optimized Standalone Version"""

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import asyncio
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Settings(BaseModel):
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

settings = Settings()

# Request Models
class BaseRequest(BaseModel):
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class QuantumAnalysisRequest(BaseRequest):
    case_type: str
    description: str
    jurisdiction: str = "NSW"
    arguments: List[str]
    precedents: Optional[List[str]] = []

class PredictionRequest(BaseRequest):
    case_data: Dict[str, Any]
    prediction_type: str = "outcome"

class StrategyRequest(BaseRequest):
    case_summary: str
    objectives: List[str]
    risk_tolerance: str = "medium"

class SearchRequest(BaseRequest):
    query: str
    search_type: str = "semantic"
    limit: int = 10

# Response Models
class BaseResponse(BaseModel):
    success: bool = True
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AnalysisResponse(BaseResponse):
    analysis_type: str
    results: Dict[str, Any]
    confidence: float

class PredictionResponse(BaseResponse):
    prediction_type: str
    prediction: Any
    confidence: float
    factors: List[Dict[str, float]]

class StrategyResponse(BaseResponse):
    strategies: List[Dict[str, Any]]
    recommended_strategy: str
    risk_assessment: Dict[str, float]

class SearchResponse(BaseResponse):
    query: str
    total_results: int
    results: List[Dict[str, Any]]

# Service Classes
class LegalRAG:
    def __init__(self):
        logger.info("Initializing Legal RAG")
    
    async def search(self, query: str, **kwargs) -> Dict:
        return {
            "total": 10,
            "results": [{"case_name": "Example Case", "relevance": 0.95}]
        }

class QuantumPredictor:
    def __init__(self):
        logger.info("Initializing Quantum Predictor")
    
    async def analyze(self, **kwargs) -> Dict:
        success_prob = 50 + len(kwargs.get('arguments', [])) * 5
        return {
            "success_probability": min(success_prob, 95),
            "overall_confidence": 0.85
        }

class MonteCarloSimulator:
    def __init__(self):
        logger.info("Initializing Monte Carlo Simulator")
    
    async def simulate(self, case_data: Dict, num_simulations: int = 1000) -> Dict:
        outcomes = np.random.choice(
            ["Plaintiff success", "Defendant success", "Settlement"],
            size=num_simulations,
            p=[0.6, 0.3, 0.1]
        )
        unique, counts = np.unique(outcomes, return_counts=True)
        probs = dict(zip(unique, counts / num_simulations))
        
        return {
            "most_likely_outcome": max(probs, key=probs.get),
            "confidence": max(probs.values()),
            "key_factors": [{"name": "Case strength", "weight": 0.4}]
        }

class StrategyEngine:
    def __init__(self):
        logger.info("Initializing Strategy Engine")
    
    async def generate(self, **kwargs) -> Dict:
        return {
            "strategies": [
                {"name": "Litigation", "probability_of_success": 0.65},
                {"name": "Settlement", "probability_of_success": 0.85}
            ],
            "recommended": "Settlement",
            "risks": {"legal": 0.3, "financial": 0.25}
        }

# Global instances
legal_rag = None
quantum = None
monte_carlo = None
strategy = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global legal_rag, quantum, monte_carlo, strategy
    
    logger.info("Starting Legal AI System...")
    legal_rag = LegalRAG()
    quantum = QuantumPredictor()
    monte_carlo = MonteCarloSimulator()
    strategy = StrategyEngine()
    
    print(f"\n{'='*60}")
    print("🏛️  AUSTRALIAN LEGAL AI - READY")
    print(f"{'='*60}")
    print(f"📍 API Docs: http://localhost:{settings.PORT}/docs")
    print(f"{'='*60}\n")
    
    yield
    
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Australian Legal AI API",
    version=settings.API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints
@app.get("/")
async def root():
    return {
        "name": "Australian Legal AI API",
        "version": settings.API_VERSION,
        "docs": f"http://localhost:{settings.PORT}/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/analysis/quantum", response_model=AnalysisResponse)
async def analyze_quantum(request: QuantumAnalysisRequest):
    result = await quantum.analyze(
        case_type=request.case_type,
        arguments=request.arguments
    )
    return AnalysisResponse(
        analysis_type="quantum_prediction",
        results=result,
        confidence=result.get("overall_confidence", 0.85)
    )

@app.post("/api/v1/prediction/simulate", response_model=PredictionResponse)
async def simulate_outcome(request: PredictionRequest):
    result = await monte_carlo.simulate(request.case_data)
    return PredictionResponse(
        prediction_type="monte_carlo_simulation",
        prediction=result["most_likely_outcome"],
        confidence=result["confidence"],
        factors=result["key_factors"]
    )

@app.post("/api/v1/strategy/generate", response_model=StrategyResponse)
async def generate_strategy(request: StrategyRequest):
    result = await strategy.generate(
        case_summary=request.case_summary,
        objectives=request.objectives
    )
    return StrategyResponse(
        strategies=result["strategies"],
        recommended_strategy=result["recommended"],
        risk_assessment=result["risks"]
    )

@app.post("/api/v1/search/cases", response_model=SearchResponse)
async def search_cases(request: SearchRequest):
    result = await legal_rag.search(request.query)
    return SearchResponse(
        query=request.query,
        total_results=result["total"],
        results=result["results"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
MAINEOF

# Create requirements.txt
cat > requirements.txt << 'REQEOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
numpy==1.24.3
websockets==11.0.3
python-multipart==0.0.6
httpx==0.25.0
REQEOF

# Create run script
cat > run_api.sh << 'RUNEOF'
#!/bin/bash
echo "🚀 Starting Optimized Legal AI API..."

# Check Python
python3 --version

# Install dependencies
pip install -r requirements.txt

# Fix numpy in existing files
for file in ultimate_intelligent_legal_api.py ultimate_legal_ai_ultra.py; do
    if [ -f "$file" ] && ! grep -q "import numpy" "$file"; then
        sed -i '1s/^/import numpy as np\n/' "$file"
        echo "✅ Fixed numpy in $file"
    fi
done

# Run the API
echo "Starting API server..."
python3 optimized_main.py
RUNEOF

chmod +x run_api.sh

# Create test script
cat > test_api.sh << 'TESTEOF'
#!/bin/bash
echo "Testing API endpoints..."

# Health check
echo "1. Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool

# Quantum analysis
echo -e "\n2. Quantum analysis:"
curl -s -X POST http://localhost:8000/api/v1/analysis/quantum \
  -H "Content-Type: application/json" \
  -d '{
    "case_type": "employment",
    "description": "Test case",
    "arguments": ["Arg1", "Arg2", "Arg3"]
  }' | python3 -m json.tool

# Monte Carlo
echo -e "\n3. Monte Carlo simulation:"
curl -s -X POST http://localhost:8000/api/v1/prediction/simulate \
  -H "Content-Type: application/json" \
  -d '{"case_data": {"type": "test"}}' | python3 -m json.tool
TESTEOF

chmod +x test_api.sh

echo "✅ Setup complete!"
echo ""
echo "To start the API:"
echo "  ./run_api.sh"
echo ""
echo "To test (in another terminal):"
echo "  ./test_api.sh"
echo ""
echo "API Documentation will be at:"
echo "  http://localhost:8000/docs"