#!/usr/bin/env python3
"""Comprehensive Test Suite for Legal AI Data Usability"""

import pytest
import asyncio
import json
import numpy as np
from typing import Dict, List, Any
import time
from datetime import datetime
import pandas as pd
from pathlib import Path

class TestDataUsability:
    """Test suite to ensure data is usable and valuable"""
    
    @pytest.fixture
    async def legal_corpus(self):
        """Load legal corpus for testing"""
        corpus_path = Path("data/legal_corpus.json")
        if not corpus_path.exists():
            # Create sample corpus for testing
            sample_corpus = self._create_sample_corpus()
            corpus_path.parent.mkdir(exist_ok=True)
            with open(corpus_path, 'w') as f:
                json.dump(sample_corpus, f)
        
        with open(corpus_path, 'r') as f:
            return json.load(f)
    
    def _create_sample_corpus(self) -> List[Dict]:
        """Create sample legal corpus for testing"""
        return [
            {
                "id": f"case_{i}",
                "case_name": f"Party A v Party B [{2020+i}]",
                "jurisdiction": np.random.choice(["NSW", "VIC", "QLD", "Federal"]),
                "court": np.random.choice(["High Court", "Federal Court", "Supreme Court"]),
                "date": f"{2020+i%5}-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}",
                "judge": f"Justice {np.random.choice(['Smith', 'Brown', 'Wilson', 'Taylor'])}",
                "legal_issues": ["contract", "negligence", "statutory interpretation"][i%3],
                "outcome": np.random.choice(["Allowed", "Dismissed", "Settled"]),
                "reasoning": f"Legal reasoning for case {i}",
                "precedents_cited": [f"Previous Case [{2015+j}]" for j in range(np.random.randint(1,4))],
                "legislation_referenced": [f"Act {j} s {np.random.randint(1,100)}" for j in range(np.random.randint(0,3))]
            }
            for i in range(100)
        ]
    
    @pytest.mark.asyncio
    async def test_data_completeness(self, legal_corpus):
        """Test that data has all required fields"""
        required_fields = [
            'case_name', 'jurisdiction', 'court', 'date',
            'legal_issues', 'outcome'
        ]
        
        incomplete_cases = []
        for case in legal_corpus:
            missing = [f for f in required_fields if not case.get(f)]
            if missing:
                incomplete_cases.append({
                    'id': case.get('id', 'unknown'),
                    'missing_fields': missing
                })
        
        assert len(incomplete_cases) == 0, f"Found {len(incomplete_cases)} incomplete cases"
    
    @pytest.mark.asyncio
    async def test_data_validity(self, legal_corpus):
        """Test data validity and consistency"""
        valid_jurisdictions = {"NSW", "VIC", "QLD", "WA", "SA", "TAS", "NT", "ACT", "Federal"}
        valid_outcomes = {"Allowed", "Dismissed", "Settled", "Remitted", "Varied"}
        
        invalid_cases = []
        
        for case in legal_corpus:
            errors = []
            
            # Check jurisdiction
            if case.get('jurisdiction') not in valid_jurisdictions:
                errors.append(f"Invalid jurisdiction: {case.get('jurisdiction')}")
            
            # Check outcome
            if case.get('outcome') not in valid_outcomes:
                errors.append(f"Invalid outcome: {case.get('outcome')}")
            
            # Check date format
            try:
                datetime.strptime(case.get('date', ''), '%Y-%m-%d')
            except ValueError:
                errors.append(f"Invalid date format: {case.get('date')}")
            
            if errors:
                invalid_cases.append({
                    'id': case.get('id'),
                    'errors': errors
                })
        
        assert len(invalid_cases) == 0, f"Found {len(invalid_cases)} invalid cases"
    
    @pytest.mark.asyncio
    async def test_data_uniqueness(self, legal_corpus):
        """Test for duplicate cases"""
        case_hashes = set()
        duplicates = []
        
        for case in legal_corpus:
            # Create hash from key fields
            hash_str = f"{case.get('case_name')}|{case.get('date')}|{case.get('court')}"
            case_hash = hash(hash_str)
            
            if case_hash in case_hashes:
                duplicates.append(case.get('id'))
            else:
                case_hashes.add(case_hash)
        
        assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate cases"
    
    @pytest.mark.asyncio
    async def test_citation_graph_connectivity(self, legal_corpus):
        """Test that citation network is well-connected"""
        import networkx as nx
        
        # Build citation graph
        G = nx.DiGraph()
        
        for case in legal_corpus:
            case_id = case.get('id')
            G.add_node(case_id)
            
            for precedent in case.get('precedents_cited', []):
                # Extract year from precedent
                import re
                year_match = re.search(r'\[(\d{4})\]', precedent)
                if year_match:
                    year = int(year_match.group(1))
                    # Find cases from that year
                    for other_case in legal_corpus:
                        if str(year) in other_case.get('date', ''):
                            G.add_edge(case_id, other_case.get('id'))
        
        # Check connectivity metrics
        if len(G) > 0:
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            connectivity_ratio = len(largest_cc) / len(G)
            
            assert connectivity_ratio > 0.3, f"Citation graph poorly connected: {connectivity_ratio:.2%}"
    
    @pytest.mark.asyncio
    async def test_temporal_distribution(self, legal_corpus):
        """Test that cases are well-distributed over time"""
        years = []
        
        for case in legal_corpus:
            try:
                year = int(case.get('date', '')[:4])
                years.append(year)
            except:
                pass
        
        if years:
            year_counts = pd.Series(years).value_counts()
            
            # Check for reasonable distribution
            assert len(year_counts) > 1, "All cases from same year"
            assert year_counts.std() / year_counts.mean() < 2, "Highly skewed temporal distribution"
    
    @pytest.mark.asyncio
    async def test_search_performance(self, legal_corpus):
        """Test search functionality performance"""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings
        texts = [f"{c.get('case_name')} {c.get('legal_issues')}" for c in legal_corpus]
        
        start_time = time.time()
        embeddings = model.encode(texts)
        encoding_time = time.time() - start_time
        
        # Test search speed
        query = "contract breach damages"
        query_embedding = model.encode([query])
        
        start_time = time.time()
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_k = np.argsort(similarities)[-10:][::-1]
        search_time = time.time() - start_time
        
        assert encoding_time < 30, f"Encoding too slow: {encoding_time:.2f}s"
        assert search_time < 0.1, f"Search too slow: {search_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_ml_feature_extraction(self, legal_corpus):
        """Test ML feature extraction from corpus"""
        features_extracted = 0
        
        for case in legal_corpus[:10]:  # Test sample
            features = {
                'jurisdiction_encoded': hash(case.get('jurisdiction', '')) % 10,
                'outcome_encoded': hash(case.get('outcome', '')) % 5,
                'precedent_count': len(case.get('precedents_cited', [])),
                'legislation_count': len(case.get('legislation_referenced', [])),
                'text_length': len(case.get('reasoning', '')),
                'year': int(case.get('date', '2020')[:4])
            }
            
            # Ensure all features are numeric
            assert all(isinstance(v, (int, float)) for v in features.values())
            features_extracted += 1
        
        assert features_extracted == 10, "Failed to extract features from all test cases"
    
    @pytest.mark.asyncio
    async def test_quantum_analysis_compatibility(self, legal_corpus):
        """Test data compatibility with quantum analysis"""
        quantum_ready_cases = 0
        
        for case in legal_corpus:
            # Check if case has quantum-analyzable properties
            has_uncertainty = bool(case.get('precedents_cited'))
            has_multiple_factors = len(case.get('legal_issues', '').split()) > 1
            has_outcome = bool(case.get('outcome'))
            
            if has_uncertainty and has_multiple_factors and has_outcome:
                quantum_ready_cases += 1
        
        quantum_ratio = quantum_ready_cases / len(legal_corpus)
        assert quantum_ratio > 0.7, f"Only {quantum_ratio:.2%} of cases quantum-ready"
    
    @pytest.mark.asyncio
    async def test_api_response_format(self, legal_corpus):
        """Test that data can be properly formatted for API responses"""
        for case in legal_corpus[:5]:
            api_response = {
                'case_id': case.get('id'),
                'summary': {
                    'name': case.get('case_name'),
                    'date': case.get('date'),
                    'outcome': case.get('outcome')
                },
                'analysis': {
                    'jurisdiction': case.get('jurisdiction'),
                    'legal_issues': case.get('legal_issues'),
                    'precedents': case.get('precedents_cited', [])
                }
            }
            
            # Test JSON serialization
            json_str = json.dumps(api_response)
            assert len(json_str) > 0
            
            # Test response size is reasonable
            assert len(json_str) < 10000, "API response too large"

# Performance benchmark tests
class TestPerformanceBenchmarks:
    """Benchmark tests for system performance"""
    
    @pytest.mark.benchmark
    async def test_corpus_loading_speed(self, benchmark):
        """Benchmark corpus loading speed"""
        async def load_corpus():
            with open("data/legal_corpus.json", 'r') as f:
                return json.load(f)
        
        result = benchmark(lambda: asyncio.run(load_corpus()))
        assert result is not None
    
    @pytest.mark.benchmark
    async def test_analysis_pipeline_speed(self, benchmark, legal_corpus):
        """Benchmark full analysis pipeline"""
        from quantum_legal_predictor import QuantumLegalPredictor
        
        predictor = QuantumLegalPredictor()
        test_case = legal_corpus[0] if legal_corpus else {}
        
        async def run_analysis():
            return await predictor.predict_quantum_enhanced(test_case)
        
        benchmark(lambda: asyncio.run(run_analysis()))
    
    @pytest.mark.benchmark
    async def test_cache_performance(self, benchmark):
        """Benchmark cache operations"""
        from intelligent_cache_manager import IntelligentCacheManager
        
        async def cache_operations():
            cache = IntelligentCacheManager(max_size_mb=10)
            await cache.initialize()
            
            # Write operations
            for i in range(100):
                await cache.set(f"key_{i}", f"value_{i}")
            
            # Read operations
            for i in range(100):
                await cache.get(f"key_{i}")
            
            return await cache.get_cache_stats()
        
        stats = benchmark(lambda: asyncio.run(cache_operations()))
        assert stats['hit_rate'] > 0.8

# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.integration
    async def test_end_to_end_analysis(self):
        """Test complete analysis workflow"""
        # Import all components
        from data_quality_engine import LegalDataQualityEngine
        from quantum_legal_predictor import QuantumLegalPredictor
        from intelligent_cache_manager import IntelligentCacheManager
        
        # Initialize components
        quality_engine = LegalDataQualityEngine()
        predictor = QuantumLegalPredictor()
        cache = IntelligentCacheManager()
        await cache.initialize()
        
        # Test workflow
        # 1. Check data quality
        metrics = await quality_engine.analyze_corpus_quality("data/legal_corpus.json")
        assert metrics.overall_score > 0.7
        
        # 2. Run prediction with caching
        test_case = {
            'case_name': 'Test v System',
            'jurisdiction': 'NSW',
            'legal_issues': 'contract breach'
        }
        
        # First call - cache miss
        result1 = await predictor.predict_quantum_enhanced(test_case)
        await cache.set('test_case_result', result1)
        
        # Second call - cache hit
        cached_result = await cache.get('test_case_result')
        assert cached_result is not None
        
        # Verify result quality
        assert result1.outcome_probability >= 0 and result1.outcome_probability <= 1
        assert len(result1.recommended_strategies) > 0

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, '-v', '--tb=short'])
