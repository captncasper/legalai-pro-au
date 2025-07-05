#!/bin/bash

echo "🧪 Australian Legal AI - Complete Test Suite"
echo "==========================================="
echo "Using Real Data: 254 Australian Legal Cases"
echo ""

# Run corpus unit tests first
echo "1️⃣ Testing Corpus Loading and Functionality"
python test_corpus_unit.py
CORPUS_RESULT=$?

echo ""
echo "2️⃣ Testing API Integration (if running)"

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is running - proceeding with integration tests"
    python test_api_simple.py
    API_RESULT=$?
else
    echo "⚠️  API not running - skipping integration tests"
    echo "   To test API, run: python legal_ai_supreme_au.py"
    API_RESULT=0
fi

echo ""
echo "==========================================="
echo "📊 Test Summary:"

if [ $CORPUS_RESULT -eq 0 ]; then
    echo "✅ Corpus tests: PASSED"
else
    echo "❌ Corpus tests: FAILED"
fi

if [ $API_RESULT -eq 0 ]; then
    echo "✅ API tests: PASSED (or skipped)"
else
    echo "❌ API tests: FAILED"
fi

echo ""
echo "📈 Your corpus contains:"
echo "   - 254 real Australian legal cases"
echo "   - 163 applicant losses"
echo "   - 47 settlements"
echo "   - 44 applicant wins"
echo "   - 307 precedent relationships"
echo ""

exit $((CORPUS_RESULT + API_RESULT))
