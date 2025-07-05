import numpy as np
#!/bin/bash
# Setup script for the optimized Legal AI project structure

echo "🏗️ Creating optimized Legal AI project structure..."

# Create main directories
mkdir -p app/{routers,core,services,models,utils}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docs
mkdir -p scripts

# Create __init__.py files
touch app/__init__.py
touch app/routers/__init__.py
touch app/core/__init__.py
touch app/services/__init__.py
touch app/models/__init__.py
touch app/utils/__init__.py

# Create configuration file
cat > app/core/config.py << 'EOF'
"""Configuration management for Legal AI API"""
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database/Index Settings
    RAG_INDEX_PATH: str = "./rag_index"
    FAISS_INDEX_PATH: str = "data/legal_index.faiss"
    DOCUMENTS_PATH: str = "data/legal_documents.pkl"
    
    # Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour
    MAX_CACHE_SIZE: int = 1000
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Security
    API_KEY_ENABLED: bool = False
    API_KEYS: List[str] = []
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
EOF

# Create shared models
cat > app/models/requests.py << 'EOF'
"""Request models for Legal AI API"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class BaseRequest(BaseModel):
    """Base request model"""
    request_id: Optional[str] = Field(None, description="Unique request ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AnalysisRequest(BaseRequest):
    """General analysis request"""
    case_type: str = Field(..., description="Type of legal case")
    description: str = Field(..., description="Case description")
    jurisdiction: str = Field("NSW", description="Legal jurisdiction")
    metadata: Optional[Dict[str, Any]] = None

class QuantumAnalysisRequest(AnalysisRequest):
    """Quantum success prediction request"""
    arguments: List[str] = Field(..., description="Legal arguments")
    precedents: Optional[List[str]] = Field(default_factory=list)
    evidence_strength: Optional[float] = Field(None, ge=0, le=100)

class PrecedentAnalysisRequest(BaseRequest):
    """Precedent analysis request"""
    case_description: str
    relevant_acts: Optional[List[str]] = None
    year_range: Optional[tuple[int, int]] = None
    limit: int = Field(10, ge=1, le=50)

class SettlementAnalysisRequest(BaseRequest):
    """Settlement analysis request"""
    case_type: str
    claim_amount: float = Field(..., gt=0)
    injury_severity: Optional[str] = None
    liability_admission: bool = False
    negotiation_stage: str = Field("initial", pattern="^(initial|mediation|pre_trial|trial)$")

class PredictionRequest(BaseRequest):
    """Prediction request"""
    case_data: Dict[str, Any]
    prediction_type: str = Field("outcome", pattern="^(outcome|duration|cost|settlement)$")
    confidence_required: float = Field(0.7, ge=0, le=1)

class StrategyRequest(BaseRequest):
    """Strategy generation request"""
    case_summary: str
    objectives: List[str]
    constraints: Optional[Dict[str, Any]] = None
    risk_tolerance: str = Field("medium", pattern="^(low|medium|high)$")

class SearchRequest(BaseRequest):
    """Search request"""
    query: str = Field(..., min_length=3)
    search_type: str = Field("semantic", pattern="^(semantic|keyword|hybrid)$")
    filters: Optional[Dict[str, Any]] = None
    limit: int = Field(10, ge=1, le=100)
EOF

# Create response models
cat > app/models/responses.py << 'EOF'
"""Response models for Legal AI API"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[float] = None

class AnalysisResponse(BaseResponse):
    """Analysis response"""
    analysis_type: str
    results: Dict[str, Any]
    confidence: float = Field(..., ge=0, le=1)
    explanations: Optional[List[str]] = None
    citations: Optional[List[Dict[str, str]]] = None

class PredictionResponse(BaseResponse):
    """Prediction response"""
    prediction_type: str
    prediction: Any
    confidence: float
    factors: List[Dict[str, float]]
    alternatives: Optional[List[Dict[str, Any]]] = None

class StrategyResponse(BaseResponse):
    """Strategy response"""
    strategies: List[Dict[str, Any]]
    recommended_strategy: str
    risk_assessment: Dict[str, float]
    timeline: Optional[List[Dict[str, Any]]] = None
    resources_required: Optional[Dict[str, Any]] = None

class SearchResponse(BaseResponse):
    """Search response"""
    query: str
    total_results: int
    results: List[Dict[str, Any]]
    facets: Optional[Dict[str, List[Dict[str, int]]]] = None
    suggestions: Optional[List[str]] = None

class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
EOF

# Create analysis router
cat > app/routers/analysis.py << 'EOF'
"""Analysis endpoints for Legal AI API"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from app.models.requests import (
    QuantumAnalysisRequest,
    PrecedentAnalysisRequest,
    SettlementAnalysisRequest,
    AnalysisRequest
)
from app.models.responses import AnalysisResponse, ErrorResponse
from app.core.dependencies import get_service, get_corpus_intel, get_legal_rag

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/quantum", response_model=AnalysisResponse)
async def analyze_quantum(
    request: QuantumAnalysisRequest,
    quantum_predictor = Depends(lambda: get_service("quantum"))
):
    """Perform quantum success prediction analysis"""
    try:
        result = await quantum_predictor.analyze(
            case_type=request.case_type,
            arguments=request.arguments,
            precedents=request.precedents,
            jurisdiction=request.jurisdiction
        )
        
        return AnalysisResponse(
            analysis_type="quantum_prediction",
            results=result,
            confidence=result.get("overall_confidence", 0.0),
            explanations=result.get("explanations", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Quantum analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/precedents", response_model=AnalysisResponse)
async def analyze_precedents(
    request: PrecedentAnalysisRequest,
    precedent_analyzer = Depends(lambda: get_service("precedent"))
):
    """Analyze relevant precedents"""
    try:
        result = await precedent_analyzer.analyze(
            case_description=request.case_description,
            relevant_acts=request.relevant_acts,
            year_range=request.year_range,
            limit=request.limit
        )
        
        return AnalysisResponse(
            analysis_type="precedent_analysis",
            results=result,
            confidence=result.get("relevance_score", 0.0),
            citations=result.get("precedents", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Precedent analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/settlement", response_model=AnalysisResponse)
async def analyze_settlement(
    request: SettlementAnalysisRequest,
    settlement_optimizer = Depends(lambda: get_service("settlement"))
):
    """Analyze settlement options and timing"""
    try:
        result = await settlement_optimizer.analyze(
            case_type=request.case_type,
            claim_amount=request.claim_amount,
            injury_severity=request.injury_severity,
            liability_admission=request.liability_admission,
            negotiation_stage=request.negotiation_stage
        )
        
        return AnalysisResponse(
            analysis_type="settlement_optimization",
            results=result,
            confidence=result.get("confidence", 0.0),
            explanations=result.get("recommendations", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Settlement analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/arguments", response_model=AnalysisResponse)
async def analyze_arguments(
    request: AnalysisRequest,
    argument_scorer = Depends(lambda: get_service("argument"))
):
    """Analyze argument strength"""
    try:
        result = await argument_scorer.analyze(
            case_type=request.case_type,
            description=request.description,
            jurisdiction=request.jurisdiction,
            metadata=request.metadata
        )
        
        return AnalysisResponse(
            analysis_type="argument_strength",
            results=result,
            confidence=result.get("overall_strength", 0.0) / 100,
            explanations=result.get("analysis", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Argument analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# Create prediction router
cat > app/routers/prediction.py << 'EOF'
"""Prediction endpoints for Legal AI API"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from app.models.requests import PredictionRequest
from app.models.responses import PredictionResponse
from app.core.dependencies import get_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/simulate", response_model=PredictionResponse)
async def simulate_outcome(
    request: PredictionRequest,
    monte_carlo = Depends(lambda: get_service("monte_carlo"))
):
    """Run Monte Carlo simulation for case outcome"""
    try:
        result = await monte_carlo.simulate(
            case_data=request.case_data,
            num_simulations=1000
        )
        
        return PredictionResponse(
            prediction_type="monte_carlo_simulation",
            prediction=result.get("most_likely_outcome"),
            confidence=result.get("confidence", 0.0),
            factors=result.get("key_factors", []),
            alternatives=result.get("outcome_distribution", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/outcome", response_model=PredictionResponse)
async def predict_outcome(
    request: PredictionRequest,
    quantum_predictor = Depends(lambda: get_service("quantum"))
):
    """Predict case outcome using quantum prediction"""
    try:
        result = await quantum_predictor.predict_outcome(
            case_data=request.case_data,
            confidence_threshold=request.confidence_required
        )
        
        return PredictionResponse(
            prediction_type="outcome_prediction",
            prediction=result.get("prediction"),
            confidence=result.get("confidence", 0.0),
            factors=result.get("contributing_factors", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/timeline", response_model=PredictionResponse)
async def predict_timeline(
    request: PredictionRequest,
    predictor = Depends(lambda: get_service("quantum"))
):
    """Predict case timeline"""
    try:
        result = await predictor.predict_timeline(
            case_data=request.case_data
        )
        
        return PredictionResponse(
            prediction_type="timeline_prediction",
            prediction=result.get("estimated_duration"),
            confidence=result.get("confidence", 0.0),
            factors=result.get("factors", []),
            alternatives=result.get("scenarios", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Timeline prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# Create strategy router
cat > app/routers/strategy.py << 'EOF'
"""Strategy endpoints for Legal AI API"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from app.models.requests import StrategyRequest
from app.models.responses import StrategyResponse
from app.core.dependencies import get_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/generate", response_model=StrategyResponse)
async def generate_strategy(
    request: StrategyRequest,
    strategy_engine = Depends(lambda: get_service("strategy"))
):
    """Generate legal strategy"""
    try:
        result = await strategy_engine.generate(
            case_summary=request.case_summary,
            objectives=request.objectives,
            constraints=request.constraints,
            risk_tolerance=request.risk_tolerance
        )
        
        return StrategyResponse(
            strategies=result.get("strategies", []),
            recommended_strategy=result.get("recommended", ""),
            risk_assessment=result.get("risks", {}),
            timeline=result.get("timeline", []),
            resources_required=result.get("resources", {}),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Strategy generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/adapt", response_model=StrategyResponse)
async def adapt_strategy(
    request: StrategyRequest,
    strategy_engine = Depends(lambda: get_service("strategy"))
):
    """Adapt existing strategy based on new information"""
    try:
        result = await strategy_engine.adapt(
            current_strategy=request.metadata.get("current_strategy", {}),
            new_information=request.metadata.get("new_information", {}),
            case_summary=request.case_summary
        )
        
        return StrategyResponse(
            strategies=result.get("adapted_strategies", []),
            recommended_strategy=result.get("recommended_adaptation", ""),
            risk_assessment=result.get("updated_risks", {}),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Strategy adaptation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# Create search router
cat > app/routers/search.py << 'EOF'
"""Search endpoints for Legal AI API"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from app.models.requests import SearchRequest
from app.models.responses import SearchResponse
from app.core.dependencies import get_legal_rag

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/cases", response_model=SearchResponse)
async def search_cases(
    request: SearchRequest,
    legal_rag = Depends(get_legal_rag)
):
    """Search legal cases"""
    try:
        result = await legal_rag.search(
            query=request.query,
            search_type=request.search_type,
            filters=request.filters,
            limit=request.limit
        )
        
        return SearchResponse(
            query=request.query,
            total_results=result.get("total", 0),
            results=result.get("results", []),
            facets=result.get("facets", {}),
            suggestions=result.get("suggestions", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/legislation", response_model=SearchResponse)
async def search_legislation(
    request: SearchRequest,
    legal_rag = Depends(get_legal_rag)
):
    """Search legislation"""
    try:
        # Add legislation-specific filter
        filters = request.filters or {}
        filters["document_type"] = "legislation"
        
        result = await legal_rag.search(
            query=request.query,
            search_type=request.search_type,
            filters=filters,
            limit=request.limit
        )
        
        return SearchResponse(
            query=request.query,
            total_results=result.get("total", 0),
            results=result.get("results", []),
            request_id=request.request_id
        )
    except Exception as e:
        logger.error(f"Legislation search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# Create admin router
cat > app/routers/admin.py << 'EOF'
"""Admin endpoints for Legal AI API"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging

from app.core.dependencies import get_corpus_intel, get_legal_rag
from app.models.responses import BaseResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stats", response_model=Dict[str, Any])
async def get_stats(corpus_intel = Depends(get_corpus_intel)):
    """Get system statistics"""
    try:
        return corpus_intel.get_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reindex", response_model=BaseResponse)
async def reindex_corpus(
    corpus_intel = Depends(get_corpus_intel),
    legal_rag = Depends(get_legal_rag)
):
    """Reindex the legal corpus"""
    try:
        # This would typically be an async background task
        await corpus_intel.reindex()
        await legal_rag.rebuild_index()
        
        return BaseResponse(
            success=True,
            message="Reindexing started"
        )
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/clear", response_model=BaseResponse)
async def clear_cache():
    """Clear all caches"""
    try:
        # Clear various caches
        from app.utils.cache import clear_all_caches
        clear_all_caches()
        
        return BaseResponse(
            success=True,
            message="Caches cleared"
        )
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
EOF

# Create dependencies module
cat > app/core/dependencies.py << 'EOF'
"""Dependency injection helpers"""
from fastapi import HTTPException

# These will be initialized by the main app
legal_rag = None
corpus_intel = None
services = {}

def get_legal_rag():
    """Get Legal RAG instance"""
    if not legal_rag:
        raise HTTPException(status_code=503, detail="Legal RAG not initialized")
    return legal_rag

def get_corpus_intel():
    """Get Corpus Intelligence instance"""
    if not corpus_intel:
        raise HTTPException(status_code=503, detail="Corpus Intelligence not initialized")
    return corpus_intel

def get_service(service_name: str):
    """Get service instance by name"""
    if service_name not in services:
        raise HTTPException(status_code=503, detail=f"Service {service_name} not available")
    return services[service_name]

def init_dependencies(rag, corpus, service_dict):
    """Initialize dependencies (called by main app)"""
    global legal_rag, corpus_intel, services
    legal_rag = rag
    corpus_intel = corpus
    services = service_dict
EOF

# Create placeholder service files
echo "Creating placeholder service files..."

# Legal RAG placeholder
cat > app/core/legal_rag.py << 'EOF'
"""Legal RAG implementation placeholder"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class LegalRAG:
    def __init__(self):
        logger.info("Initializing Legal RAG")
        # Initialize ChromaDB, embeddings, etc.
        
    async def query_async(self, query: str, context: Dict = None) -> Dict:
        """Async query method"""
        # Implement RAG query logic
        return {
            "answer": "Legal RAG response placeholder",
            "sources": [],
            "confidence": 0.85
        }
    
    async def search(self, query: str, **kwargs) -> Dict:
        """Search method"""
        return {
            "total": 0,
            "results": [],
            "facets": {},
            "suggestions": []
        }
    
    async def rebuild_index(self):
        """Rebuild index"""
        logger.info("Rebuilding index...")
EOF

# Corpus Intelligence placeholder
cat > app/core/corpus_intelligence.py << 'EOF'
"""Corpus Intelligence implementation placeholder"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CorpusIntelligence:
    def __init__(self):
        logger.info("Initializing Corpus Intelligence")
        self.stats = {
            "cases": 33913,
            "settlements": 47111,
            "precedents": 38796
        }
        
    async def load_corpus(self):
        """Load corpus data"""
        logger.info("Loading corpus...")
        
    def get_stats(self) -> Dict:
        """Get corpus statistics"""
        return self.stats
        
    async def reindex(self):
        """Reindex corpus"""
        logger.info("Reindexing corpus...")
EOF

# Create service placeholders
for service in quantum_predictor monte_carlo precedent_analyzer settlement_optimizer argument_scorer strategy_engine; do
    cat > app/services/${service}.py << EOF
"""${service} service implementation placeholder"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ${service}:
    def __init__(self, corpus_intel):
        self.corpus_intel = corpus_intel
        logger.info("Initializing ${service}")
        
    async def analyze(self, **kwargs) -> Dict:
        """Analyze method"""
        return {
            "status": "success",
            "results": {},
            "confidence": 0.85
        }
EOF
done

# Update service class names
sed -i 's/quantum_predictor/QuantumSuccessPredictor/g' app/services/quantum_predictor.py
sed -i 's/monte_carlo/MonteCarloSimulator/g' app/services/monte_carlo.py
sed -i 's/precedent_analyzer/PrecedentAnalyzer/g' app/services/precedent_analyzer.py
sed -i 's/settlement_optimizer/SettlementOptimizer/g' app/services/settlement_optimizer.py
sed -i 's/argument_scorer/ArgumentStrengthScorer/g' app/services/argument_scorer.py
sed -i 's/strategy_engine/StrategyEngine/g' app/services/strategy_engine.py

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pydantic-settings==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
sentence-transformers==2.2.2
chromadb==0.4.15
faiss-cpu==1.7.4
pandas==2.0.3
python-multipart==0.0.6
websockets==11.0.3
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
httpx==0.25.0
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.10.1
flake8==6.1.0
mypy==1.6.1
EOF

# Create .env template
cat > .env.example << 'EOF'
# API Configuration
API_VERSION=1.0.0
DEBUG=False
HOST=0.0.0.0
PORT=8000

# Security
API_KEY_ENABLED=False
API_KEYS=[]

# Paths
RAG_INDEX_PATH=./rag_index
FAISS_INDEX_PATH=data/legal_index.faiss
DOCUMENTS_PATH=data/legal_documents.pkl

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Cache Configuration
CACHE_TTL=3600
MAX_CACHE_SIZE=1000

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
EOF

# Create run script
cat > run_optimized.sh << 'EOF'
#!/bin/bash
# Run the optimized Legal AI API

echo "🚀 Starting Optimized Legal AI API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
fi

# Run the application
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
EOF

chmod +x run_optimized.sh

# Create migration script
cat > migrate_to_optimized.py << 'EOF'
#!/usr/bin/env python3
"""
Migration script to move from old API versions to optimized version
"""
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate():
    """Migrate existing code to optimized structure"""
    
    # Map old services to new structure
    migrations = {
        "legal_rag.py": "app/core/legal_rag.py",
        "corpus_intelligence_extractor.py": "app/core/corpus_intelligence.py",
        "next_gen_legal_ai_features.py": "app/services/",
        "src/search.py": "app/core/search.py",
        "src/embeddings.py": "app/core/embeddings.py"
    }
    
    logger.info("Starting migration to optimized structure...")
    
    for old_path, new_path in migrations.items():
        if os.path.exists(old_path):
            if os.path.isdir(new_path):
                # Extract classes and move to appropriate service files
                logger.info(f"Migrating {old_path} to service modules...")
            else:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copy2(old_path, new_path)
                logger.info(f"Migrated {old_path} -> {new_path}")
    
    logger.info("Migration complete!")
    logger.info("Next steps:")
    logger.info("1. Review the migrated code in app/")
    logger.info("2. Update imports in the service files")
    logger.info("3. Run tests to ensure everything works")

if __name__ == "__main__":
    migrate()
EOF

chmod +x migrate_to_optimized.py

echo "✅ Optimized project structure created!"
echo ""
echo "Directory structure:"
tree app -I '__pycache__'

echo ""
echo "Next steps:"
echo "1. Run ./migrate_to_optimized.py to migrate existing code"
echo "2. Update the placeholder service implementations with your actual code"
echo "3. Run ./run_optimized.sh to start the optimized API"
echo "4. Visit http://localhost:8000/docs for API documentation"