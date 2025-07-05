#!/bin/bash

echo "⚛️ Testing Quantum Legal Predictions..."

curl -X POST "$BASE_URL/api/v1/analysis/quantum-supreme" \
  -H "Content-Type: application/json" \
  -d '{
    "case_name": "Test Corp v Smart Systems",
    "jurisdiction": "nsw",
    "case_type": "contract_breach",
    "description": "Breach of AI development contract with penalty clauses",
    "arguments": {
      "plaintiff": "Failed to deliver working AI system on time",
      "defendant": "Specifications changed multiple times"
    },
    "evidence_strength": 0.75
  }' | python3 -m json.tool
