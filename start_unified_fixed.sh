#!/bin/bash

echo "🚀 Starting Unified Australian Legal AI System (Fixed)"
echo "===================================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install any missing dependencies
echo "📦 Checking dependencies..."
pip install -q fastapi uvicorn sentence-transformers numpy

# Start the fixed unified system
echo "🌐 Starting API server..."
echo ""
echo "📍 API will be available at:"
echo "   http://localhost:8000"
echo "   http://localhost:8000/docs (Interactive API docs)"
echo ""

python unified_legal_ai_system_fixed.py
