#!/usr/bin/env python3
"""Test the new alternative scrapers and upload features"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_features():
    print("🧪 Testing New Features")
    print("=" * 60)
    
    # Test 1: Alternative scrapers
    print("\n1️⃣ Testing Alternative Scrapers...")
    response = requests.post(
        f"{BASE_URL}/api/v1/scrape/alternatives",
        params={"query": "artificial intelligence", "source": "federal"}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Federal Court: Found {data['cases_found']} cases")
    else:
        print(f"❌ Alternative scraper test failed: {response.status_code}")
    
    # Test 2: Case upload
    print("\n2️⃣ Testing Case Upload...")
    test_case = {
        "citation": "[2024] TEST 001",
        "case_name": "AI Test Case v Legal System",
        "text": "This is a test case about artificial intelligence in the legal system...",
        "outcome": "applicant_won",
        "jurisdiction": "nsw",
        "court": "TEST",
        "catchwords": "ARTIFICIAL INTELLIGENCE - test case - demonstration"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/cases/upload",
        data=test_case
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Case uploaded: {data['message']}")
        print(f"   Corpus size: {data['corpus_size']}")
    else:
        print(f"❌ Case upload failed: {response.status_code}")
    
    # Test 3: Check uploaded cases
    print("\n3️⃣ Checking Uploaded Cases...")
    response = requests.get(f"{BASE_URL}/api/v1/cases/uploaded")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Total uploaded cases: {data['total']}")
        if data['cases']:
            print(f"   Latest: {data['cases'][0]['citation']}")
    
    # Test 4: Search for uploaded case
    print("\n4️⃣ Searching for Uploaded Case...")
    response = requests.post(
        f"{BASE_URL}/api/v1/search",
        json={"query": "AI Test Case", "search_type": "keyword"}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Search found {data['results_count']} results")
    
    print("\n✨ Visit http://localhost:8000/upload for the web upload interface!")

if __name__ == "__main__":
    test_features()
