#!/usr/bin/env python3
"""Test suite for Enhanced Legal AI"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, path, data=None):
    """Test an endpoint and print results"""
    print(f"\n🧪 Testing: {name}")
    
    try:
        if method == "GET":
            r = requests.get(f"{BASE_URL}{path}")
        else:
            r = requests.post(f"{BASE_URL}{path}", json=data)
        
        if r.status_code == 200:
            print(f"✅ Success!")
            result = r.json()
            if "results" in result:
                print(f"📊 Preview: {json.dumps(result['results'], indent=2)[:200]}...")
        else:
            print(f"❌ Failed: {r.status_code}")
            print(f"Error: {r.text[:100]}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Run tests
print("🚀 Testing Enhanced Legal AI API")
print("="*60)

# Basic endpoints
test_endpoint("Root", "GET", "/")
test_endpoint("Health", "GET", "/health")

# Quantum Analysis
test_endpoint("Quantum Analysis", "POST", "/api/v1/analysis/quantum", {
    "case_type": "employment",
    "description": "Wrongful termination",
    "arguments": ["No cause", "Retaliation", "Discrimination"],
    "precedents": ["Case A v B", "Case C v D"],
    "evidence_strength": 85
})

# Monte Carlo (different types)
test_endpoint("Monte Carlo - Standard", "POST", "/api/v1/prediction/simulate", {
    "case_data": {"strength_score": 75},
    "simulation_type": "standard"
})

test_endpoint("Monte Carlo - Bayesian", "POST", "/api/v1/prediction/simulate", {
    "case_data": {"strength_score": 75},
    "simulation_type": "bayesian"
})

test_endpoint("Monte Carlo - Quantum", "POST", "/api/v1/prediction/simulate", {
    "case_data": {"strength_score": 75},
    "simulation_type": "quantum"
})

# Emotion Analysis
test_endpoint("Emotion Analysis", "POST", "/api/v1/analysis/emotion", {
    "text": "I am devastated by this unfair treatment. The company's actions have caused me significant distress.",
    "context": "victim_statement"
})

# Pattern Recognition
test_endpoint("Pattern Recognition", "POST", "/api/v1/analysis/pattern", {
    "case_description": "Employee fired after reporting safety violations",
    "pattern_type": "all"
})

# Settlement Calculator
test_endpoint("Settlement Calculator", "POST", "/api/v1/calculate/settlement", {
    "case_type": "personal_injury",
    "claim_amount": 250000,
    "injury_severity": "severe",
    "liability_admission": True,
    "negotiation_stage": "mediation"
})

# Document Generation
test_endpoint("Generate Contract", "POST", "/api/v1/generate/document", {
    "document_type": "contract",
    "context": {
        "parties": ["ABC Corp", "John Smith"],
        "purpose": "Software development services",
        "duration": "6 months",
        "compensation": "$50,000"
    }
})

test_endpoint("Generate Letter", "POST", "/api/v1/generate/document", {
    "document_type": "letter",
    "context": {
        "sender_name": "Law Firm LLP",
        "recipient_name": "Mr. Jones",
        "subject": "Settlement Offer",
        "body": "We write to propose a settlement..."
    }
})

# Search
test_endpoint("Search Cases", "POST", "/api/v1/search/cases", {
    "query": "employment discrimination",
    "search_type": "hybrid",
    "filters": {"jurisdiction": "NSW", "year_from": 2020},
    "limit": 5
})

# Admin
test_endpoint("Admin Stats", "GET", "/api/v1/admin/stats")

print("\n✅ All tests completed!")
