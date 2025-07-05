#!/bin/bash

echo "📊 Testing Data Quality Engine..."

python3 << 'PYTHON'
import asyncio
import json
from data_quality_engine import LegalDataQualityEngine

async def test_quality():
    print("Analyzing legal corpus quality...")
    engine = LegalDataQualityEngine()
    
    # Create test corpus
    test_corpus = [
        {
            "id": "test_case_1",
            "case_name": "Smith v Jones [2024]",
            "jurisdiction": "NSW",
            "court": "Supreme Court",
            "date": "2024-03-15",
            "judge": "Justice Brown",
            "legal_issues": "contract breach damages",
            "outcome": "Allowed",
            "reasoning": "Clear breach established with documented damages",
            "precedents_cited": ["Brown v Green [2023]", "White v Black [2022]"],
            "legislation_referenced": ["Contracts Act 1999 s 45"]
        }
    ]
    
    # Save test corpus
    with open("test_corpus.json", "w") as f:
        json.dump(test_corpus, f)
    
    # Analyze
    metrics = await engine.analyze_corpus_quality("test_corpus.json")
    
    print(f"\n📈 Data Quality Metrics:")
    print(f"  Overall Score: {metrics.overall_score:.2%}")
    print(f"  Completeness: {metrics.completeness:.2%}")
    print(f"  Consistency: {metrics.consistency:.2%}")
    print(f"  Accuracy: {metrics.accuracy:.2%}")
    print(f"  Validity: {metrics.validity:.2%}")
    
    if metrics.recommendations:
        print("\n💡 Recommendations:")
        for rec in metrics.recommendations[:3]:
            print(f"  • {rec}")

asyncio.run(test_quality())
PYTHON
