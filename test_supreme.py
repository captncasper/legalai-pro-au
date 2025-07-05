#!/usr/bin/env python3
"""Test suite for Australian Legal AI SUPREME"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, path, data=None):
    print(f"\n🧪 Testing: {name}")
    try:
        if method == "GET":
            r = requests.get(f"{BASE_URL}{path}")
        else:
            r = requests.post(f"{BASE_URL}{path}", json=data)
        
        if r.status_code == 200:
            print(f"✅ Success!")
            return r.json()
        else:
            print(f"❌ Failed: {r.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

print("🇦🇺 AUSTRALIAN LEGAL AI SUPREME - Test Suite")
print("=" * 60)

# Test endpoints
test_endpoint("Root", "GET", "/")
test_endpoint("Health", "GET", "/health")

# Test quantum analysis
quantum_data = {
    "case_type": "employment",
    "description": "Unfair dismissal case",
    "jurisdiction": "nsw",
    "arguments": ["No warnings", "Good performance", "Retaliation"],
    "evidence_strength": 85,
    "damages_claimed": 150000
}

result = test_endpoint("Quantum Analysis", "POST", "/api/v1/analysis/quantum-supreme", quantum_data)
if result and "analysis" in result:
    print(f"   Success Probability: {result['analysis']['success_probability']}%")
    print(f"   Confidence: {result['analysis']['confidence_level']}")

test_endpoint("Admin Stats", "GET", "/api/v1/admin/stats")

print("\n✅ Tests completed!")
