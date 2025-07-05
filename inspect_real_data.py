#!/usr/bin/env python3
"""Inspect actual corpus data to build real tests"""

import json
import pickle
import gzip
import os
from pathlib import Path

def inspect_corpus_files():
    """Check all corpus files and their structure"""
    
    print("🔍 Inspecting Australian Legal Corpus Files...")
    print("=" * 60)
    
    # Check corpus_intelligence.json
    if Path("corpus_intelligence.json").exists():
        with open("corpus_intelligence.json", "r") as f:
            corpus_intel = json.load(f)
        print(f"\n📊 corpus_intelligence.json:")
        print(f"   - Type: {type(corpus_intel)}")
        
        if isinstance(corpus_intel, dict):
            print(f"   - Top-level keys: {list(corpus_intel.keys())}")
            print(f"   - Total top-level entries: {len(corpus_intel)}")
            
            # Inspect each section
            for key, value in corpus_intel.items():
                print(f"\n   📁 '{key}' section:")
                print(f"      - Type: {type(value)}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"      - Count: {len(value)} items")
                    # Show structure of first item
                    first_item = value[0]
                    if isinstance(first_item, dict):
                        print(f"      - Item keys: {list(first_item.keys())}")
                        # Show sample values
                        for k, v in list(first_item.items())[:3]:
                            if isinstance(v, str):
                                preview = v[:50] + "..." if len(v) > 50 else v
                            else:
                                preview = str(v)[:50]
                            print(f"        • {k}: {preview}")
                    else:
                        print(f"      - Item type: {type(first_item)}")
                elif isinstance(value, dict):
                    print(f"      - Sub-keys: {list(value.keys())[:5]}...")
    
    # Check demo_compressed_corpus.pkl.gz
    if Path("demo_compressed_corpus.pkl.gz").exists():
        print(f"\n📊 demo_compressed_corpus.pkl.gz:")
        try:
            with gzip.open("demo_compressed_corpus.pkl.gz", "rb") as f:
                demo_corpus = pickle.load(f)
            print(f"   - Type: {type(demo_corpus)}")
            
            if isinstance(demo_corpus, list):
                print(f"   - Total items: {len(demo_corpus)}")
                if demo_corpus:
                    first_item = demo_corpus[0]
                    if isinstance(first_item, dict):
                        print(f"   - Item keys: {list(first_item.keys())}")
                        for k, v in list(first_item.items())[:3]:
                            preview = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
                            print(f"     • {k}: {preview}")
            elif isinstance(demo_corpus, dict):
                print(f"   - Keys: {list(demo_corpus.keys())[:5]}...")
        except Exception as e:
            print(f"   - Error loading: {e}")
    
    # Check data directory
    if Path("data").exists():
        print(f"\n📂 data/ directory contents:")
        data_files = list(Path("data").iterdir())
        if data_files:
            for file in sorted(data_files)[:10]:  # Show first 10
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"   - {file.name}: {size_mb:.2f} MB")
                
                # Sample JSON files
                if file.suffix == '.json' and size_mb < 10:
                    try:
                        with open(file, 'r') as f:
                            sample_data = json.load(f)
                        if isinstance(sample_data, list) and sample_data:
                            print(f"     Type: list of {len(sample_data)} items")
                            if isinstance(sample_data[0], dict):
                                print(f"     Keys: {list(sample_data[0].keys())[:5]}...")
                        elif isinstance(sample_data, dict):
                            print(f"     Type: dict with {len(sample_data)} keys")
                            print(f"     Keys: {list(sample_data.keys())[:5]}...")
                    except:
                        pass
        else:
            print("   (No files found)")
    
    # Check hf_extracted_intelligence.json structure
    if Path("hf_extracted_intelligence.json").exists():
        print(f"\n📊 hf_extracted_intelligence.json (Large file):")
        file_size = Path("hf_extracted_intelligence.json").stat().st_size / 1024 / 1024
        print(f"   - Size: {file_size:.1f} MB")
        
        # Just peek at structure due to size
        with open("hf_extracted_intelligence.json", "r") as f:
            # Read first line to check if it's array or object
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                print("   - Type: JSON array")
                # Read until we find first complete object
                buffer = ""
                bracket_count = 0
                for char in f.read(5000):
                    buffer += char
                    if char == '{':
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            break
                
                try:
                    # Parse first item
                    first_item_str = buffer[1:].strip().rstrip(',')
                    if first_item_str.endswith('}'):
                        first_item = json.loads(first_item_str)
                        print(f"   - First item keys: {list(first_item.keys())[:10]}...")
                except:
                    print("   - Could not parse first item")
            else:
                print("   - Type: JSON object")
    
    # Check corpus_upload.zip
    if Path("corpus_upload.zip").exists():
        size_mb = Path("corpus_upload.zip").stat().st_size / 1024 / 1024
        print(f"\n📦 corpus_upload.zip: {size_mb:.1f} MB")
        
    # Look for other potential corpus files
    print(f"\n🔎 Other potential corpus files:")
    for pattern in ['*.json', '*.pkl', '*.csv', '*.txt']:
        files = list(Path('.').glob(pattern))
        for file in files[:5]:  # Limit to 5 per pattern
            if file.name not in ['corpus_intelligence.json', 'hf_extracted_intelligence.json']:
                size_mb = file.stat().st_size / 1024 / 1024
                if size_mb > 0.1:  # Only show files > 100KB
                    print(f"   - {file.name}: {size_mb:.2f} MB")

if __name__ == "__main__":
    inspect_corpus_files()
