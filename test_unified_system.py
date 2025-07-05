#!/usr/bin/env python3
"""Test the unified system"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_system():
    print("🧪 Testing Unified Legal AI System")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1️⃣ Health Check:")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test 2: Root endpoint
    print("\n2️⃣ System Info:")
    response = requests.get(f"{BASE_URL}/")
    data = response.json()
    print(f"   System: {data['system']}")
    print(f"   Features: {', '.join(data['features'])}")
    print(f"   Total cases: {data['corpus_stats']['total_cases']}")
    
    # Test 3: Semantic search
    print("\n3️⃣ Semantic Search Test:")
    response = requests.post(f"{BASE_URL}/api/v1/search", json={
        "query": "negligence personal injury",
        "search_type": "semantic",
        "limit": 3
    })
    results = response.json()
    print(f"   Found {results['results_count']} results")
    for r in results['results'][:2]:
        print(f"   - {r['case_name']} ({r['year']})")
        print(f"     Similarity: {r['similarity_score']:.2f}")
        if r['settlement_amount']:
            print(f"     Settlement: {r['settlement_amount']}")
    
    # Test 4: Prediction
    print("\n4️⃣ Outcome Prediction Test:")
    response = requests.post(f"{BASE_URL}/api/v1/predict", json={
        "case_description": "Slip and fall accident in shopping center causing serious injury",
        "jurisdiction": "nsw",
        "case_type": "negligence",
        "evidence_strength": 0.8
    })
    prediction = response.json()
    pred = prediction['prediction']
    print(f"   Predicted outcome: {pred['predicted_outcome']}")
    print(f"   Win probability: {pred['applicant_wins']:.1%}")
    print(f"   Settlement probability: {pred['settles']:.1%}")
    print(f"   Loss probability: {pred['applicant_loses']:.1%}")
    print(f"   Confidence: {prediction['confidence']:.1%}")
    
    # Test 5: Comprehensive analysis
    print("\n5️⃣ Comprehensive Analysis Test:")
    response = requests.post(f"{BASE_URL}/api/v1/analyze", json={
        "case_name": "Test v Shopping Center",
        "description": "Slip and fall with permanent injury, clear negligence",
        "jurisdiction": "nsw"
    })
    analysis = response.json()
    print(f"   Predicted outcome: {analysis['prediction']['predicted_outcome']}")
    if analysis['settlement_analysis']['average']:
        print(f"   Expected settlement: ${analysis['settlement_analysis']['average']:,.0f}")
        print(f"   Settlement range: ${analysis['settlement_analysis']['range']['min']:,.0f} - ${analysis['settlement_analysis']['range']['max']:,.0f}")
    
    # Test 6: Statistics
    print("\n6️⃣ Corpus Statistics:")
    response = requests.get(f"{BASE_URL}/api/v1/statistics")
    stats = response.json()
    print(f"   Total cases: {stats['total_cases']}")
    print(f"   Outcome distribution:")
    for outcome, count in stats['outcome_distribution'].items():
        print(f"     - {outcome}: {count} cases")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    # Wait a moment for server to be ready
    print("Waiting for server to start...")
    time.sleep(2)
    
    try:
        test_system()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on port 8000")
