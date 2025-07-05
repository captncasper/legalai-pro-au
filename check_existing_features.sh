#!/bin/bash

echo "🔍 Analyzing your existing smart features..."
echo "==========================================="

# Check legal_ai_mega.py structure
echo -e "\n📁 legal_ai_mega.py structure:"
grep -n "^class\|^def\|async def" legal_ai_mega.py | head -20

# Check if it has FastAPI
echo -e "\n🌐 API setup in legal_ai_mega.py:"
grep -n "FastAPI\|@app\|@router" legal_ai_mega.py | head -10

# Check semantic search implementation
echo -e "\n🔍 Semantic search features:"
grep -n "semantic\|embedding\|SentenceTransformer" legal_ai_mega.py | head -10

# Check prediction features
echo -e "\n🔮 Prediction features:"
grep -n "predict\|quantum\|probability" legal_ai_mega.py | head -10

# Check what corpus format it expects
echo -e "\n📚 Data loading in legal_ai_mega.py:"
grep -n "load.*corpus\|load.*data" legal_ai_mega.py | head -10

