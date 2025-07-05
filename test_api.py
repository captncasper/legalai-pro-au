import numpy as np
#!/usr/bin/env python3
"""Simple test script for the optimized API"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_quantum_analysis():
    """Test quantum analysis"""
    data = {
        "case_type": "employment",
        "description": "Wrongful termination case",
        "jurisdiction": "NSW",
        "arguments": [
            "Employee terminated without cause",
            "Excellent performance reviews",
            "Discrimination suspected"
        ]
    }
    response = requests.post(f"{BASE_URL}/api/v1/analysis/quantum", json=data)
    print("\nQuantum Analysis:", json.dumps(response.json(), indent=2))

def test_monte_carlo():
    """Test Monte Carlo simulation"""
    data = {
        "case_data": {
            "case_type": "employment",
            "strength_score": 75,
            "precedent_support": 80
        },
        "prediction_type": "outcome"
    }
    response = requests.post(f"{BASE_URL}/api/v1/prediction/simulate", json=data)
    print("\nMonte Carlo Simulation:", json.dumps(response.json(), indent=2))

def test_strategy():
    """Test strategy generation"""
    data = {
        "case_summary": "Employment dispute with strong evidence",
        "objectives": ["Maximize compensation", "Quick resolution"],
        "risk_tolerance": "medium"
    }
    response = requests.post(f"{BASE_URL}/api/v1/strategy/generate", json=data)
    print("\nStrategy Generation:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Testing Optimized Legal AI API...")
    print("="*50)
    
    try:
        test_health()
        test_quantum_analysis()
        test_monte_carlo()
        test_strategy()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the API is running on http://localhost:8000")
