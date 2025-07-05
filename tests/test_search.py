"""Test search functionality"""
import pytest
from src.search import LegalSearchEngine


def test_search_returns_results():
    # Test that search returns expected number of results
    engine = LegalSearchEngine()
    results = engine.search("contract requirements", k=3)
    assert len(results) == 3
    assert all('document' in r for r in results)
    assert all('relevance_score' in r for r in results)
