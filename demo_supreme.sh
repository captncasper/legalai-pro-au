#!/bin/bash
# Australian Legal AI SUPREME - Demo Script

echo "🇦🇺 AUSTRALIAN LEGAL AI SUPREME - DEMO"
echo "======================================"
echo ""
echo "This demo will showcase the key features of the most advanced"
echo "legal AI system in Australia."
echo ""
echo "Press Enter to continue..."
read

# 1. Quantum Analysis Demo
echo -e "\n1️⃣ QUANTUM SUPREME ANALYSIS - NSW Employment Case"
echo "================================================"
curl -s -X POST http://localhost:8000/api/v1/analysis/quantum-supreme \
  -H "Content-Type: application/json" \
  -d '{
    "case_type": "employment",
    "description": "Senior manager terminated after 10 years without warning",
    "jurisdiction": "nsw",
    "arguments": [
      "No performance issues documented",
      "Termination immediately after raising concerns about financial irregularities",
      "Similar positions advertised next day",
      "No consultation or warning provided"
    ],
    "precedents": ["Byrne v Australian Airlines"],
    "evidence_strength": 88,
    "damages_claimed": 250000
  }' | python3 -m json.tool | head -50

echo -e "\n\nPress Enter for next demo..."
read

# 2. AI Judge Demo
echo -e "\n2️⃣ AI JUDGE SYSTEM - Predicting Judicial Decision"
echo "================================================"
curl -s -X POST http://localhost:8000/api/v1/analysis/ai-judge \
  -H "Content-Type: application/json" \
  -d '{
    "case_summary": "Unfair dismissal with whistleblower elements",
    "plaintiff_arguments": [
      "Termination was harsh and unreasonable",
      "Protected disclosure made",
      "Exemplary employment record"
    ],
    "defendant_arguments": [
      "Genuine redundancy",
      "Business restructure"
    ],
    "evidence_presented": [
      {"type": "document", "description": "Email raising concerns"},
      {"type": "witness", "description": "Colleagues confirming disclosure"}
    ],
    "applicable_laws": ["Fair Work Act 2009"],
    "precedents_cited": ["Byrne v Australian Airlines"],
    "jurisdiction": "federal"
  }' | python3 -m json.tool | head -40

echo -e "\n\nPress Enter for next demo..."
read

# 3. Contract Analysis Demo
echo -e "\n3️⃣ CONTRACT ANALYSIS - Risk Assessment"
echo "====================================="
curl -s -X POST http://localhost:8000/api/v1/analysis/contract \
  -H "Content-Type: application/json" \
  -d '{
    "contract_text": "1. Services: Consulting services\n2. Payment: $100,000\n3. Liability: Unlimited liability for all losses\n4. Termination: At will by either party",
    "contract_type": "service_agreement",
    "party_position": "first_party",
    "risk_tolerance": "low",
    "jurisdiction": "nsw"
  }' | python3 -m json.tool | head -40

echo -e "\n\nPress Enter for next demo..."
read

# 4. Legal Research Demo
echo -e "\n4️⃣ LEGAL RESEARCH - Comprehensive Search"
echo "======================================="
curl -s -X POST http://localhost:8000/api/v1/research/comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "research_query": "whistleblower protection employment termination",
    "research_depth": "comprehensive",
    "case_law_years": 5,
    "jurisdiction": "federal"
  }' | python3 -m json.tool | head -40

echo -e "\n\n✅ Demo completed!"
echo "For full API documentation, visit: http://localhost:8000/docs"
