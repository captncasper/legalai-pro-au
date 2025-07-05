#!/usr/bin/env python3
"""Find and extract real case data from corpus files"""

import json
import pickle
import gzip
from pathlib import Path
import re

def find_case_data():
    """Search all files for actual case data"""
    
    print("🔍 Searching for Real Case Data...")
    print("=" * 60)
    
    all_cases = []
    
    # 1. Check corpus_intelligence.json for case data
    if Path("corpus_intelligence.json").exists():
        with open("corpus_intelligence.json", "r") as f:
            corpus_intel = json.load(f)
        
        # Check case_outcomes section
        if 'case_outcomes' in corpus_intel:
            print(f"\n✅ Found case_outcomes section")
            outcomes = corpus_intel['case_outcomes']
            if isinstance(outcomes, list):
                print(f"   - {len(outcomes)} case outcomes")
                all_cases.extend(outcomes)
                # Show sample
                if outcomes:
                    print(f"   - Sample: {outcomes[0]}")
    
    # 2. Check data directory for case files
    if Path("data").exists():
        print(f"\n📂 Checking data directory...")
        for json_file in Path("data").glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if it contains cases
                if isinstance(data, list) and data:
                    # Check if items look like cases
                    first_item = data[0]
                    if isinstance(first_item, dict):
                        # Look for case-like fields
                        case_fields = ['case_name', 'citation', 'court', 'judgment', 'title', 'parties']
                        if any(field in first_item for field in case_fields):
                            print(f"\n✅ Found cases in {json_file.name}")
                            print(f"   - {len(data)} cases")
                            all_cases.extend(data)
                            
                            # Show structure
                            print(f"   - Keys: {list(first_item.keys())[:10]}")
            except Exception as e:
                pass
    
    # 3. Check compressed corpus
    if Path("demo_compressed_corpus.pkl.gz").exists():
        print(f"\n📦 Checking compressed corpus...")
        try:
            with gzip.open("demo_compressed_corpus.pkl.gz", "rb") as f:
                compressed_data = pickle.load(f)
            
            if isinstance(compressed_data, list):
                print(f"   - {len(compressed_data)} items in compressed corpus")
                if compressed_data and isinstance(compressed_data[0], dict):
                    all_cases.extend(compressed_data)
            elif isinstance(compressed_data, dict):
                # Check if dict contains cases
                for key, value in compressed_data.items():
                    if isinstance(value, list) and value:
                        if isinstance(value[0], dict) and any(k in value[0] for k in ['case_name', 'citation']):
                            print(f"   - Found {len(value)} cases under key '{key}'")
                            all_cases.extend(value)
        except Exception as e:
            print(f"   - Error: {e}")
    
    # 4. Look for cases in large JSON file
    if Path("hf_extracted_intelligence.json").exists():
        print(f"\n📊 Sampling hf_extracted_intelligence.json...")
        # Due to size, just check structure
        with open("hf_extracted_intelligence.json", "r") as f:
            # Read first 10000 chars
            sample = f.read(10000)
            
            # Look for case patterns
            case_patterns = [
                r'"case_name":\s*"([^"]+)"',
                r'"citation":\s*"([^"]+)"',
                r'"title":\s*"([^"]+v[^"]+)"',  # Looks for "v" in title
            ]
            
            for pattern in case_patterns:
                matches = re.findall(pattern, sample)
                if matches:
                    print(f"   - Found case data with pattern: {pattern}")
                    print(f"   - Sample matches: {matches[:3]}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"Total case-like records found: {len(all_cases)}")
    
    if all_cases:
        # Analyze structure
        print(f"\n🔧 Case Data Structure Analysis:")
        
        # Find common keys
        all_keys = set()
        for case in all_cases[:100]:  # Sample first 100
            if isinstance(case, dict):
                all_keys.update(case.keys())
        
        print(f"Common fields found: {sorted(all_keys)}")
        
        # Show sample cases
        print(f"\n📋 Sample Cases:")
        for i, case in enumerate(all_cases[:3]):
            print(f"\nCase {i+1}:")
            if isinstance(case, dict):
                for key, value in list(case.items())[:5]:
                    if isinstance(value, str):
                        preview = value[:100] + "..." if len(value) > 100 else value
                    else:
                        preview = str(value)[:100]
                    print(f"  {key}: {preview}")
        
        # Save sample for testing
        with open('sample_cases.json', 'w') as f:
            json.dump(all_cases[:10], f, indent=2)
        print(f"\n💾 Saved 10 sample cases to sample_cases.json")

if __name__ == "__main__":
    find_case_data()
