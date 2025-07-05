#!/usr/bin/env python3
"""
Feature Integration Helper - Adds features from existing files to enhanced API
"""

import os
import ast
import re

def extract_features_from_file(filename):
    """Extract useful features from existing API files"""
    if not os.path.exists(filename):
        return None
    
    print(f"\n📄 Analyzing {filename}...")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract classes
    classes = re.findall(r'class\s+(\w+).*?:\n((?:\s{4}.*\n)*)', content)
    
    # Extract endpoints
    endpoints = re.findall(r'@app\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)["\']', content)
    
    # Extract interesting functions
    functions = re.findall(r'async def (\w+)\s*\([^)]*\).*?:', content)
    
    return {
        'classes': [c[0] for c in classes],
        'endpoints': endpoints,
        'functions': functions
    }

# Analyze existing files
files_to_analyze = [
    'ultimate_intelligent_legal_api.py',
    'ultimate_legal_ai_supreme.py',
    'ultimate_legal_ai_ultra.py',
    'next_gen_legal_ai_features.py'
]

all_features = {}

for file in files_to_analyze:
    features = extract_features_from_file(file)
    if features:
        all_features[file] = features
        print(f"  Found: {len(features['classes'])} classes, {len(features['endpoints'])} endpoints")

# Generate integration report
print("\n📊 Feature Integration Report")
print("="*60)

all_classes = set()
all_endpoints = set()

for file, features in all_features.items():
    all_classes.update(features['classes'])
    all_endpoints.update([(e[0], e[1]) for e in features['endpoints']])

print(f"\n✨ Unique features found:")
print(f"  - Classes: {len(all_classes)}")
print(f"  - Endpoints: {len(all_endpoints)}")

print("\n🎯 Key features to integrate:")
priority_features = [
    "QuantumSuccessPredictor",
    "PatternRecognitionEngine", 
    "RiskAnalysisEngine",
    "CollaborationHub",
    "VoiceCommandProcessor"
]

for feature in priority_features:
    if feature in all_classes:
        print(f"  ✅ {feature}")
    else:
        print(f"  ❌ {feature} (not found)")

print("\n💡 Next steps:")
print("1. The enhanced version already includes core features")
print("2. Run: ./migrate_to_enhanced.sh to upgrade")
print("3. Test with: ./test_enhanced.py")
print("4. Add more features by editing legal_ai_enhanced.py")
