#!/bin/bash

echo "🚀 Starting Unified Australian Legal AI System"
echo "============================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install any missing dependencies
echo "📦 Checking dependencies..."
pip install -q fastapi uvicorn sentence-transformers numpy

# Check if embeddings exist
if [ ! -f "case_embeddings.pkl" ]; then
    echo "📊 Creating case embeddings (first time only)..."
    python -c "from unified_legal_ai_system import unified_ai; print('Embeddings created')"
fi

# Start the unified system
echo "🌐 Starting API server..."
echo ""
echo "📍 API will be available at:"
echo "   http://localhost:8000"
echo "   http://localhost:8000/docs (Interactive API docs)"
echo ""
echo "✨ Features available:"
echo "   - Semantic Search (/api/v1/search)"
echo "   - Case Outcome Prediction (/api/v1/predict)"
echo "   - Comprehensive Analysis (/api/v1/analyze)"
echo "   - Judge Analysis (/api/v1/judge/{name})"
echo "   - Corpus Statistics (/api/v1/statistics)"
echo "   - WebSocket support (/ws)"
echo ""

python unified_legal_ai_system.py
