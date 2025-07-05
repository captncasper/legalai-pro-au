#!/usr/bin/env python3
"""Load real Australian legal corpus from your actual files"""

import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

class AustralianLegalCorpus:
    """Load and manage real Australian legal cases"""
    
    def __init__(self):
        self.cases = []
        self.case_outcomes = []
        self.precedent_network = []
        self.judge_patterns = {}
        self.winning_patterns = {}
        self.corpus_loaded = False
        
    def load_corpus(self):
        """Load all corpus data from your files"""
        print("📚 Loading Australian Legal Corpus...")
        
        # Load corpus_intelligence.json
        if Path("corpus_intelligence.json").exists():
            with open("corpus_intelligence.json", "r") as f:
                corpus_intel = json.load(f)
            
            # Load case outcomes
            if 'case_outcomes' in corpus_intel:
                self.case_outcomes = corpus_intel['case_outcomes']
                print(f"✅ Loaded {len(self.case_outcomes)} case outcomes")
                
                # Convert to standard case format
                for outcome_data in self.case_outcomes:
                    case = {
                        'citation': outcome_data.get('citation', ''),
                        'outcome': outcome_data.get('outcome', ''),
                        'text': outcome_data.get('text_sample', ''),
                        'factors': outcome_data.get('factors', []),
                        'case_name': self._extract_case_name(outcome_data.get('citation', '')),
                        'year': self._extract_year(outcome_data.get('citation', '')),
                        'court': self._extract_court(outcome_data.get('text_sample', ''))
                    }
                    self.cases.append(case)
            
            # Load other intelligence
            if 'precedent_network' in corpus_intel:
                self.precedent_network = corpus_intel['precedent_network']
                print(f"✅ Loaded {len(self.precedent_network)} precedent relationships")
            
            if 'judge_patterns' in corpus_intel:
                self.judge_patterns = corpus_intel['judge_patterns']
                print(f"✅ Loaded patterns for {len(self.judge_patterns)} judges")
            
            if 'winning_patterns' in corpus_intel:
                self.winning_patterns = corpus_intel['winning_patterns']
                print(f"✅ Loaded winning patterns")
        
        # Load compressed corpus
        if Path("demo_compressed_corpus.pkl.gz").exists():
            try:
                with gzip.open("demo_compressed_corpus.pkl.gz", "rb") as f:
                    compressed_data = pickle.load(f)
                
                if isinstance(compressed_data, dict) and 'documents' in compressed_data:
                    print(f"✅ Found compressed documents")
            except Exception as e:
                print(f"⚠️  Could not load compressed corpus: {e}")
        
        self.corpus_loaded = True
        print(f"\n📊 Total cases loaded: {len(self.cases)}")
        
    def _extract_case_name(self, citation: str) -> str:
        """Extract case name from citation"""
        # Pattern: "Party v Party [Year]"
        match = re.match(r'^([^[]+)\s*\[', citation)
        return match.group(1).strip() if match else citation
    
    def _extract_year(self, citation: str) -> int:
        """Extract year from citation"""
        match = re.search(r'\[(\d{4})\]', citation)
        return int(match.group(1)) if match else 0
    
    def _extract_court(self, text: str) -> str:
        """Extract court from text sample"""
        text_lower = text.lower()
        
        # Common Australian courts
        courts = {
            'high court': 'HCA',
            'federal court': 'FCA',
            'federal circuit': 'FCCA',
            'supreme court': 'SC',
            'district court': 'DC',
            'magistrates court': 'MC',
            'land and environment': 'LEC',
            'industrial relations': 'IRC'
        }
        
        for court_name, abbreviation in courts.items():
            if court_name in text_lower:
                # Try to find state
                states = ['nsw', 'vic', 'qld', 'wa', 'sa', 'tas', 'act', 'nt']
                for state in states:
                    if state in text_lower:
                        return f"{state.upper()}{abbreviation}"
                return abbreviation
        
        return 'Unknown'
    
    def search_cases(self, query: str, filters: Dict[str, Any] = None) -> List[Dict]:
        """Search cases using real data"""
        if not self.corpus_loaded:
            self.load_corpus()
        
        results = []
        query_lower = query.lower()
        
        for case in self.cases:
            # Search in citation, case name, and text
            searchable = f"{case['citation']} {case['case_name']} {case['text']}".lower()
            
            if query_lower in searchable:
                # Apply filters if provided
                if filters:
                    if 'outcome' in filters and case['outcome'] != filters['outcome']:
                        continue
                    if 'year' in filters and case['year'] != filters['year']:
                        continue
                    if 'court' in filters and filters['court'] not in case['court']:
                        continue
                
                results.append(case)
                
                if len(results) >= 20:  # Limit results
                    break
        
        return results
    
    def get_case_by_citation(self, citation: str) -> Optional[Dict]:
        """Get specific case by citation"""
        if not self.corpus_loaded:
            self.load_corpus()
        
        for case in self.cases:
            if case['citation'] == citation:
                return case
        return None
    
    def get_judge_statistics(self, judge_name: str) -> Dict:
        """Get real judge statistics"""
        if judge_name.upper() in self.judge_patterns:
            return self.judge_patterns[judge_name.upper()]
        return {}
    
    def get_precedent_network_for_case(self, citation: str) -> List[Dict]:
        """Get precedent relationships for a case"""
        relationships = []
        
        for rel in self.precedent_network:
            if rel['citing'] == citation or rel['cited'] == citation:
                relationships.append(rel)
        
        return relationships
    
    def get_outcome_distribution(self) -> Dict[str, int]:
        """Get distribution of case outcomes"""
        distribution = {}
        
        for case in self.case_outcomes:
            outcome = case.get('outcome', 'unknown')
            distribution[outcome] = distribution.get(outcome, 0) + 1
        
        return distribution

# Global instance
corpus = AustralianLegalCorpus()

if __name__ == "__main__":
    # Test loading
    corpus.load_corpus()
    
    # Show some statistics
    print("\n📊 Corpus Statistics:")
    outcomes = corpus.get_outcome_distribution()
    print(f"Outcome distribution:")
    for outcome, count in sorted(outcomes.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {outcome}: {count} cases")
    
    # Test search
    print("\n🔍 Testing search:")
    results = corpus.search_cases("negligence")
    print(f"Found {len(results)} cases mentioning 'negligence'")
    if results:
        print(f"First result: {results[0]['citation']}")
