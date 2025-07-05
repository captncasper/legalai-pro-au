#!/bin/bash
echo "Testing API endpoints..."

# Health check
echo "1. Health check:"
curl -s http://localhost:8000/health | python3 -m json.tool

# Quantum analysis
echo -e "\n2. Quantum analysis:"
curl -s -X POST http://localhost:8000/api/v1/analysis/quantum \
  -H "Content-Type: application/json" \
  -d '{
    "case_type": "employment",
    "description": "Test case",
    "arguments": ["Arg1", "Arg2", "Arg3"]
  }' | python3 -m json.tool

# Monte Carlo
echo -e "\n3. Monte Carlo simulation:"
curl -s -X POST http://localhost:8000/api/v1/prediction/simulate \
  -H "Content-Type: application/json" \
  -d '{"case_data": {"type": "test"}}' | python3 -m json.tool
