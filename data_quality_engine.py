#!/usr/bin/env python3
"""Advanced Data Quality Engine for Legal Corpus"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiofiles
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.cluster import DBSCAN
from collections import defaultdict
import hashlib

@dataclass
class DataQualityMetrics:
    completeness: float
    consistency: float
    accuracy: float
    timeliness: float
    uniqueness: float
    validity: float
    overall_score: float
    issues: List[Dict[str, Any]]
    recommendations: List[str]

class LegalDataQualityEngine:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.legal_entities = self._load_legal_entities()
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.legal_bert = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
        
    async def analyze_corpus_quality(self, corpus_path: str) -> DataQualityMetrics:
        """Comprehensive corpus quality analysis"""
        corpus_data = await self._load_corpus(corpus_path)
        
        # Run parallel quality checks
        tasks = [
            self._check_completeness(corpus_data),
            self._check_consistency(corpus_data),
            self._check_accuracy(corpus_data),
            self._check_timeliness(corpus_data),
            self._check_uniqueness(corpus_data),
            self._check_validity(corpus_data)
        ]
        
        results = await asyncio.gather(*tasks)
        
        metrics = DataQualityMetrics(
            completeness=results[0][0],
            consistency=results[1][0],
            accuracy=results[2][0],
            timeliness=results[3][0],
            uniqueness=results[4][0],
            validity=results[5][0],
            overall_score=np.mean([r[0] for r in results]),
            issues=self._aggregate_issues(results),
            recommendations=self._generate_recommendations(results)
        )
        
        return metrics
    
    async def _check_completeness(self, data: List[Dict]) -> Tuple[float, List[Dict]]:
        """Check for missing fields and incomplete data"""
        required_fields = [
            'case_name', 'jurisdiction', 'date', 'court',
            'judge', 'parties', 'legal_issues', 'outcome',
            'reasoning', 'precedents_cited', 'legislation_referenced'
        ]
        
        issues = []
        complete_count = 0
        
        for idx, case in enumerate(data):
            missing_fields = [f for f in required_fields if not case.get(f)]
            if not missing_fields:
                complete_count += 1
            else:
                issues.append({
                    'case_id': case.get('id', idx),
                    'type': 'incomplete',
                    'missing_fields': missing_fields
                })
        
        completeness_score = complete_count / len(data) if data else 0
        return completeness_score, issues
    
    async def _check_consistency(self, data: List[Dict]) -> Tuple[float, List[Dict]]:
        """Check for inconsistent data patterns"""
        issues = []
        consistency_checks = 0
        passed_checks = 0
        
        # Date format consistency
        date_formats = defaultdict(int)
        for case in data:
            if date_str := case.get('date'):
                format_type = self._detect_date_format(date_str)
                date_formats[format_type] += 1
                consistency_checks += 1
        
        # Most common format should be > 90%
        if date_formats:
            max_format_count = max(date_formats.values())
            if max_format_count / sum(date_formats.values()) > 0.9:
                passed_checks += 1
            else:
                issues.append({
                    'type': 'inconsistent_date_formats',
                    'formats_found': dict(date_formats)
                })
        
        # Jurisdiction naming consistency
        jurisdictions = defaultdict(list)
        for case in data:
            if juris := case.get('jurisdiction'):
                normalized = self._normalize_jurisdiction(juris)
                if normalized != juris:
                    jurisdictions[normalized].append(juris)
                    consistency_checks += 1
        
        for norm, variants in jurisdictions.items():
            if len(set(variants)) > 1:
                issues.append({
                    'type': 'inconsistent_jurisdiction_names',
                    'normalized': norm,
                    'variants': list(set(variants))
                })
            else:
                passed_checks += 1
        
        consistency_score = passed_checks / consistency_checks if consistency_checks else 1
        return consistency_score, issues
    
    async def _check_accuracy(self, data: List[Dict]) -> Tuple[float, List[Dict]]:
        """Check for legal accuracy and citation validity"""
        issues = []
        accuracy_checks = 0
        passed_checks = 0
        
        for case in data[:100]:  # Sample for performance
            # Validate case citations
            if citations := case.get('precedents_cited', []):
                for citation in citations:
                    accuracy_checks += 1
                    if self._validate_citation_format(citation):
                        passed_checks += 1
                    else:
                        issues.append({
                            'case_id': case.get('id'),
                            'type': 'invalid_citation',
                            'citation': citation
                        })
            
            # Validate legislation references
            if legislation := case.get('legislation_referenced', []):
                for leg in legislation:
                    accuracy_checks += 1
                    if self._validate_legislation_reference(leg):
                        passed_checks += 1
                    else:
                        issues.append({
                            'case_id': case.get('id'),
                            'type': 'invalid_legislation_reference',
                            'reference': leg
                        })
        
        accuracy_score = passed_checks / accuracy_checks if accuracy_checks else 1
        return accuracy_score, issues
    
    async def _check_timeliness(self, data: List[Dict]) -> Tuple[float, List[Dict]]:
        """Check data currency and update patterns"""
        issues = []
        current_year = datetime.now().year
        recent_cases = 0
        
        for case in data:
            if date_str := case.get('date'):
                try:
                    case_year = self._extract_year(date_str)
                    if current_year - case_year <= 2:
                        recent_cases += 1
                    elif current_year - case_year > 10:
                        issues.append({
                            'case_id': case.get('id'),
                            'type': 'outdated',
                            'year': case_year
                        })
                except:
                    issues.append({
                        'case_id': case.get('id'),
                        'type': 'unparseable_date',
                        'date': date_str
                    })
        
        timeliness_score = recent_cases / len(data) if data else 0
        return timeliness_score, issues
    
    async def _check_uniqueness(self, data: List[Dict]) -> Tuple[float, List[Dict]]:
        """Check for duplicate cases using multiple methods"""
        issues = []
        
        # Method 1: Exact hash matching
        case_hashes = {}
        exact_duplicates = 0
        
        for idx, case in enumerate(data):
            case_hash = self._generate_case_hash(case)
            if case_hash in case_hashes:
                exact_duplicates += 1
                issues.append({
                    'type': 'exact_duplicate',
                    'case_id': case.get('id', idx),
                    'duplicate_of': case_hashes[case_hash]
                })
            else:
                case_hashes[case_hash] = case.get('id', idx)
        
        # Method 2: Semantic similarity for near-duplicates
        if len(data) < 1000:  # Only for smaller datasets
            embeddings = self._generate_case_embeddings(data[:100])
            clustering = DBSCAN(eps=0.1, min_samples=2, metric='cosine')
            clusters = clustering.fit_predict(embeddings)
            
            for cluster_id in set(clusters):
                if cluster_id != -1:  # Not noise
                    cluster_cases = [i for i, c in enumerate(clusters) if c == cluster_id]
                    if len(cluster_cases) > 1:
                        issues.append({
                            'type': 'semantic_duplicate_cluster',
                            'case_ids': [data[i].get('id', i) for i in cluster_cases]
                        })
        
        uniqueness_score = 1 - (exact_duplicates / len(data)) if data else 1
        return uniqueness_score, issues
    
    async def _check_validity(self, data: List[Dict]) -> Tuple[float, List[Dict]]:
        """Check legal validity of case data"""
        issues = []
        valid_cases = 0
        
        for case in data:
            case_valid = True
            
            # Check court hierarchy validity
            if court := case.get('court'):
                if jurisdiction := case.get('jurisdiction'):
                    if not self._validate_court_jurisdiction(court, jurisdiction):
                        case_valid = False
                        issues.append({
                            'case_id': case.get('id'),
                            'type': 'invalid_court_jurisdiction',
                            'court': court,
                            'jurisdiction': jurisdiction
                        })
            
            # Check outcome validity
            if outcome := case.get('outcome'):
                if not self._validate_legal_outcome(outcome):
                    case_valid = False
                    issues.append({
                        'case_id': case.get('id'),
                        'type': 'invalid_outcome',
                        'outcome': outcome
                    })
            
            if case_valid:
                valid_cases += 1
        
        validity_score = valid_cases / len(data) if data else 1
        return validity_score, issues
    
    def _generate_recommendations(self, results: List[Tuple]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for score, issues in results:
            if score < 0.8:
                if any(i['type'] == 'incomplete' for i in issues):
                    recommendations.append(
                        "Implement automated field extraction from original documents"
                    )
                if any(i['type'] == 'inconsistent_date_formats' for i in issues):
                    recommendations.append(
                        "Standardize date formats using ISO 8601 (YYYY-MM-DD)"
                    )
                if any(i['type'] == 'exact_duplicate' for i in issues):
                    recommendations.append(
                        "Implement deduplication pipeline before corpus ingestion"
                    )
                if any(i['type'] == 'outdated' for i in issues):
                    recommendations.append(
                        "Schedule regular corpus updates from recent case databases"
                    )
        
        return list(set(recommendations))
    
    # Helper methods
    def _load_legal_entities(self):
        return {
            'courts': set(['High Court', 'Federal Court', 'Supreme Court', ...]),
            'jurisdictions': set(['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'NT', 'ACT', 'Federal']),
            'valid_outcomes': set(['Allowed', 'Dismissed', 'Settled', 'Remitted', ...])
        }
    
    def _normalize_jurisdiction(self, jurisdiction: str) -> str:
        mapping = {
            'New South Wales': 'NSW',
            'Victoria': 'VIC',
            'Queensland': 'QLD',
            # ... etc
        }
        return mapping.get(jurisdiction, jurisdiction)
    
    def _validate_citation_format(self, citation: str) -> bool:
        # Australian citation format: Party v Party [Year] Court Number
        import re
        pattern = r'.+ v .+ \[\d{4}\] \w+ \d+'
        return bool(re.match(pattern, citation))
    
    def _generate_case_hash(self, case: Dict) -> str:
        key_fields = ['case_name', 'date', 'court', 'jurisdiction']
        content = '|'.join(str(case.get(f, '')) for f in key_fields)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_case_embeddings(self, cases: List[Dict]) -> np.ndarray:
        texts = []
        for case in cases:
            text = f"{case.get('case_name', '')} {case.get('legal_issues', '')} {case.get('outcome', '')}"
            texts.append(text)
        return self.embedder.encode(texts)

# Create test suite
if __name__ == "__main__":
    async def test_data_quality():
        engine = LegalDataQualityEngine()
        metrics = await engine.analyze_corpus_quality("data/legal_corpus.json")
        
        print(f"Overall Data Quality Score: {metrics.overall_score:.2%}")
        print(f"Completeness: {metrics.completeness:.2%}")
        print(f"Consistency: {metrics.consistency:.2%}")
        print(f"Accuracy: {metrics.accuracy:.2%}")
        print(f"Timeliness: {metrics.timeliness:.2%}")
        print(f"Uniqueness: {metrics.uniqueness:.2%}")
        print(f"Validity: {metrics.validity:.2%}")
        
        print("\nTop Issues:")
        for issue in metrics.issues[:10]:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        for rec in metrics.recommendations:
            print(f"  • {rec}")
    
    asyncio.run(test_data_quality())
