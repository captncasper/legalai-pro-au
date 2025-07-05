#!/bin/bash
# Test API endpoints

echo "🧪 Testing Australian Legal AI API..."

# Base URL detection
if [ -n "$CODESPACES" ]; then
    BASE_URL="https://$CODESPACE_NAME-8000.preview.app.github.dev"
else
    BASE_URL="http://localhost:8000"
fi

echo "Using base URL: $BASE_URL"
echo ""

# Test root endpoint
echo "1. Testing root endpoint..."
curl -s "$BASE_URL/" | python -m json.tool

echo ""
echo "2. Testing search endpoint..."
curl -s -X POST "$BASE_URL/search" \
  -H "Authorization: Bearer demo_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "contract requirements", "num_results": 3}' \
  | python -m json.tool

echo ""
echo "✅ API tests complete!"
