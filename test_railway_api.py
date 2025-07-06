#!/usr/bin/env python3
"""Test the Railway deployment API to see formatted outputs"""
import requests
import json
from datetime import datetime

# Your Railway URL
BASE_URL = "https://legalai-pro-au-production.up.railway.app"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 50)

def test_legal_brief():
    """Test legal brief generation"""
    print("\n📄 Testing Legal Brief Generation...")
    
    test_data = {
        "matter_type": "negligence",
        "client_name": "John Smith",
        "opposing_party": "Woolworths Group Limited",
        "jurisdiction": "NSW",
        "court_level": "District Court",
        "case_facts": "Client slipped on wet floor at Woolworths Bondi Junction on 15 March 2024. No warning signs were present. Client suffered broken ankle requiring surgery and 6 weeks off work.",
        "legal_issues": "Duty of care breach, failure to warn of hazard, premises liability",
        "damages_sought": 75000,
        "brief_type": "statement_of_claim"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/generate-legal-brief",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                brief = data["legal_brief"]
                
                print("\n" + "="*60)
                print("GENERATED LEGAL BRIEF")
                print("="*60)
                
                # Document Header
                print(f"\n{brief['document_header']['title']}")
                print(f"{brief['document_header']['court']}")
                print(f"{brief['document_header']['matter_details']}")
                
                # Parties
                print("\n--- PARTIES ---")
                print(f"Plaintiff: {brief['parties']['plaintiff']}")
                print(f"Defendant: {brief['parties']['defendant']}")
                
                # Facts
                print("\n--- STATEMENT OF FACTS ---")
                for i, fact in enumerate(brief['statement_of_facts'], 1):
                    print(f"{i}. {fact}")
                
                # Legal Issues
                print("\n--- LEGAL ISSUES ---")
                for i, issue in enumerate(brief['legal_issues'], 1):
                    print(f"{i}. {issue}")
                
                # Damages
                print("\n--- DAMAGES ---")
                if 'damages_breakdown' in brief:
                    for category, details in brief['damages_breakdown'].items():
                        print(f"\n{category}:")
                        for item, value in details.items():
                            print(f"  - {item}: ${value:,.2f}")
                else:
                    print("Damages structure:", brief.get('damages', 'Not found'))
                
                if 'total_claim' in brief:
                    print(f"\nTotal Claim: ${brief['total_claim']:,.2f}")
                else:
                    print(f"\nDamages Sought: ${test_data['damages_sought']:,.2f}")
                
                print("\n" + "="*60)
                
                # Also print raw JSON for debugging
                print("\n--- RAW JSON STRUCTURE ---")
                print(json.dumps(brief, indent=2)[:1000] + "...")
                
            else:
                print(f"Error: {data}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 50)

def test_case_search():
    """Test enhanced case search with RAG functionality"""
    print("\n🔍 Testing Enhanced Case Search with RAG...")
    
    test_queries = [
        "slip and fall woolworths negligence",
        "employment unfair dismissal",
        "contract breach construction",
        "personal injury motor vehicle accident"
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        
        test_query = {"query": query}
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/search",
                json=test_query,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                search_enhanced = data.get("search_enhanced", False)
                
                print(f"Search Enhanced (RAG): {search_enhanced}")
                print(f"Found {len(results)} results")
                
                for i, result in enumerate(results[:2], 1):  # Show first 2
                    print(f"\n  Result {i}:")
                    print(f"    Title: {result.get('title', 'N/A')}")
                    print(f"    Score: {result.get('score', 0):.3f}")
                    print(f"    Type: {result.get('type', 'N/A')}")
                    search_type = result.get('search_type', result.get('rag_score', 'unknown'))
                    print(f"    Search Type: {search_type}")
                    snippet = result.get('text', result.get('snippet', ''))[:150]
                    print(f"    Snippet: {snippet}...")
                    
            else:
                print(f"Error Response: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("-" * 50)

def test_case_outcome_prediction():
    """Test case outcome prediction"""
    print("\n🔮 Testing Case Outcome Prediction...")
    
    test_data = {
        "case_type": "negligence",
        "facts": "Elderly customer slipped on unmarked wet floor in supermarket, suffered hip fracture",
        "jurisdiction": "NSW",
        "opposing_party_type": "corporation",
        "claim_amount": 150000,
        "evidence_strength": "strong"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/predict-case-outcome",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            prediction = data.get("prediction", {})
            
            print("\n--- Prediction Results ---")
            print(f"Success Probability: {prediction.get('success_probability', 0)}%")
            print(f"Settlement Likely: {prediction.get('settlement_likely', False)}")
            print(f"Estimated Settlement: ${prediction.get('estimated_settlement_range', {}).get('mid', 0):,.2f}")
            print(f"Time to Resolution: {prediction.get('estimated_duration', 'N/A')}")
            
            print("\n--- Risk Factors ---")
            for factor in prediction.get('risk_factors', []):
                print(f"  • {factor}")
                
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("-" * 50)

if __name__ == "__main__":
    print(f"🧪 Testing Railway API at: {BASE_URL}")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run all tests
    test_health()
    test_legal_brief()
    test_case_search()
    test_case_outcome_prediction()
    
    print("\n✅ All tests completed!")