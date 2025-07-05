#!/bin/bash

echo "🧪 Australian Legal AI - Real Data Test Suite"
echo "==========================================="

# Check API
echo "Checking if API is running..."
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ API not running. Starting it now..."
    python legal_ai_supreme_au.py &
    sleep 5
fi

# Run tests
echo -e "\n1️⃣ Loading Real Corpus"
python load_real_aussie_corpus.py

echo -e "\n2️⃣ Running Integration Tests"
python test_with_real_data.py

# Show results
echo -e "\n✅ Test run complete!"
echo "Check the output above for detailed results."
