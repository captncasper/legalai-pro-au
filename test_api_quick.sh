#!/bin/bash
if [ -n "$CODESPACES" ]; then
    URL="https://${CODESPACE_NAME}-8000.preview.app.github.dev"
else
    URL="http://localhost:8000"
fi

echo "Testing API at: $URL"
curl -X POST "$URL/search" \
  -H "Authorization: Bearer demo_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Fair Work Act employment law", "num_results": 3}' | python3 -m json.tool
