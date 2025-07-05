#!/bin/bash
echo "🧪 Testing Legal AI API..."

# Health check
echo -e "\n1. Health Check:"
curl -s http://localhost:8000/health | python3 -m json.tool || echo "API not responding"

# Quantum analysis
echo -e "\n2. Quantum Analysis:"
curl -s -X POST http://localhost:8000/api/v1/analysis/quantum \
  -H "Content-Type: application/json" \
  -d '{
    "case_type": "employment",
    "description": "Test case",
    "arguments": ["Arg1", "Arg2", "Arg3"]
  }' | python3 -m json.tool || echo "Quantum endpoint failed"

# Monte Carlo
echo -e "\n3. Monte Carlo:"
curl -s -X POST http://localhost:8000/api/v1/prediction/simulate \
  -H "Content-Type: application/json" \
  -d '{"case_data": {"type": "test"}}' | python3 -m json.tool || echo "Monte Carlo failed"

# Search
echo -e "\n4. Search:"
curl -s -X POST http://localhost:8000/api/v1/search/cases \
  -H "Content-Type: application/json" \
  -d '{"query": "employment law", "limit": 5}' | python3 -m json.tool || echo "Search failed"

echo -e "\n✅ Test complete!"
