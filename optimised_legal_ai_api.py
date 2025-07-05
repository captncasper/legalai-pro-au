#!/usr/bin/env python3
"""
Australian Legal AI API - Optimized Consolidated Version
Combines all features from multiple versions into a clean, modular architecture
"""

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import asyncio
from functools import lru_cache

# Import routers (we'll create these next)
from app.routers import analysis, prediction, strategy, search, admin
from app.core.config import settings
from app.core.legal_rag import LegalRAG
from app.core.corpus_intelligence import CorpusIntelligence
from app.services.quantum_predictor import QuantumSuccessPredictor
from app.services.monte_carlo import MonteCarloSimulator
from app.services.precedent_analyzer import PrecedentAnalyzer
from app.services.settlement_optimizer import SettlementOptimizer
from app.services.argument_scorer import ArgumentStrengthScorer
from app.services.strategy_engine import StrategyEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
legal_rag = None
corpus_intel = None
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("🚀 Starting Australian Legal AI System...")
    
    # Initialize core components
    global legal_rag, corpus_intel, services
    
    try:
        # Initialize RAG system
        legal_rag = LegalRAG()
        logger.info("✅ Legal RAG initialized")
        
        # Initialize corpus intelligence
        corpus_intel = CorpusIntelligence()
        await corpus_intel.load_corpus()
        logger.info(f"✅ Corpus loaded: {corpus_intel.get_stats()}")
        
        # Initialize services
        services = {
            'quantum': QuantumSuccessPredictor(corpus_intel),
            'monte_carlo': MonteCarloSimulator(corpus_intel),
            'precedent': PrecedentAnalyzer(legal_rag, corpus_intel),
            'settlement': SettlementOptimizer(corpus_intel),
            'argument': ArgumentStrengthScorer(legal_rag),
            'strategy': StrategyEngine(corpus_intel)
        }
        logger.info("✅ All services initialized")
        
        # Print startup banner
        print_startup_banner(corpus_intel.get_stats())
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
        
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Australian Legal AI System...")
    # Add cleanup code here if needed

def print_startup_banner(stats: Dict):
    """Print startup banner with system stats"""
    banner = f"""
{'='*60}
🏛️  AUSTRALIAN LEGAL AI SYSTEM - OPTIMIZED
{'='*60}
✅ Corpus Intelligence: {stats.get('cases', 0):,} cases
✅ Settlement Data: {stats.get('settlements', 0):,} amounts  
✅ Precedent Network: {stats.get('precedents', 0):,} precedents
✅ AI Engines: ACTIVE
✅ API Version: {settings.API_VERSION}
{'='*60}
🌐 API Docs: http://localhost:{settings.PORT}/docs
🔧 Health: http://localhost:{settings.PORT}/health
{'='*60}
    """
    print(banner)

# Create FastAPI app
app = FastAPI(
    title="Australian Legal AI API",
    description="Advanced AI-powered legal analysis system for Australian law",
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(prediction.router, prefix="/api/v1/prediction", tags=["Prediction"])
app.include_router(strategy.router, prefix="/api/v1/strategy", tags=["Strategy"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Australian Legal AI API",
        "version": settings.API_VERSION,
        "status": "operational",
        "documentation": f"http://localhost:{settings.PORT}/docs",
        "endpoints": {
            "analysis": "/api/v1/analysis",
            "prediction": "/api/v1/prediction", 
            "strategy": "/api/v1/strategy",
            "search": "/api/v1/search",
            "admin": "/api/v1/admin"
        }
    }

# Health check endpoint
@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        stats = corpus_intel.get_stats() if corpus_intel else {}
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "rag": legal_rag is not None,
                "corpus": corpus_intel is not None,
                "quantum": 'quantum' in services,
                "monte_carlo": 'monte_carlo' in services,
                "precedent": 'precedent' in services,
                "settlement": 'settlement' in services,
                "argument": 'argument' in services,
                "strategy": 'strategy' in services
            },
            "corpus_stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# WebSocket for real-time legal assistant
@app.websocket("/ws/assistant")
async def websocket_assistant(websocket: WebSocket):
    """WebSocket endpoint for real-time legal assistant"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            query = data.get("query", "")
            context = data.get("context", {})
            
            # Process query
            try:
                # Use RAG for intelligent response
                response = await legal_rag.query_async(query, context)
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "data": response,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc) if settings.DEBUG else "An error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Dependency injection helpers
def get_legal_rag() -> LegalRAG:
    """Get Legal RAG instance"""
    if not legal_rag:
        raise HTTPException(status_code=503, detail="Legal RAG not initialized")
    return legal_rag

def get_corpus_intel() -> CorpusIntelligence:
    """Get Corpus Intelligence instance"""
    if not corpus_intel:
        raise HTTPException(status_code=503, detail="Corpus Intelligence not initialized")
    return corpus_intel

def get_service(service_name: str):
    """Get service instance by name"""
    if service_name not in services:
        raise HTTPException(status_code=503, detail=f"Service {service_name} not available")
    return services[service_name]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )