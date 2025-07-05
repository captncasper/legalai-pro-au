#!/usr/bin/env python3
"""
Corpus Intelligence Extractor
Learns patterns, outcomes, and strategies from your legal corpus
"""

import pickle
from collections import defaultdict, Counter
import re
from typing import Dict, List, Tuple
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

class CorpusIntelligenceExtractor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.patterns_learned = {
            'winning_cases': defaultdict(list),
            'losing_cases': defaultdict(list),
            'settlement_amounts': [],
            'time_to_resolution': [],
            'judge_patterns': defaultdict(list),
            'precedent_chains': []
        }
    
    def extract_intelligence(self, documents: List[Dict]) -> Dict:
        """Extract deep intelligence from corpus"""
        
        print("🧠 Extracting intelligence from corpus...")
        
        # 1. Extract case outcomes
        case_outcomes = self._extract_case_outcomes(documents)
        print(f"✅ Found {len(case_outcomes)} cases with clear outcomes")
        
        # 2. Learn winning patterns
        winning_patterns = self._learn_winning_patterns(case_outcomes)
        print(f"✅ Identified {len(winning_patterns)} winning patterns")
        
        # 3. Extract settlement ranges
        settlement_data = self._extract_settlement_amounts(documents)
        print(f"✅ Found {len(settlement_data)} settlement amounts")
        
        # 4. Build precedent network
        precedent_network = self._build_precedent_network(documents)
        print(f"✅ Built network with {len(precedent_network)} precedent chains")
        
        # 5. Extract judge behavior patterns
        judge_patterns = self._analyze_judge_patterns(documents)
        print(f"✅ Analyzed {len(judge_patterns)} judges")
        
        # 6. Learn temporal patterns
        temporal_patterns = self._learn_temporal_patterns(documents)
        
        # 7. Create embedding clusters
        case_clusters = self._create_case_clusters(documents[:1000])  # Sample for speed
        
        return {
            'case_outcomes': case_outcomes,
            'winning_patterns': winning_patterns,
            'settlement_intelligence': settlement_data,
            'precedent_network': precedent_network,
            'judge_patterns': judge_patterns,
            'temporal_patterns': temporal_patterns,
            'case_clusters': case_clusters,
            'corpus_insights': self._generate_insights(case_outcomes, winning_patterns)
        }
    
    def _extract_case_outcomes(self, documents: List[Dict]) -> List[Dict]:
        """Extract cases with clear outcomes"""
        outcomes = []
        
        outcome_patterns = {
            'applicant_won': [
                r'application\s+granted',
                r'found\s+in\s+favor\s+of\s+applicant',
                r'unfair\s+dismissal\s+established',
                r'compensation\s+ordered'
            ],
            'applicant_lost': [
                r'application\s+dismissed',
                r'found\s+against\s+applicant',
                r'dismissal\s+was\s+fair',
                r'no\s+jurisdiction'
            ],
            'settled': [
                r'matter\s+settled',
                r'consent\s+order',
                r'parties\s+reached\s+agreement'
            ]
        }
        
        for doc in documents:
            text = doc.get('text', '').lower()
            citation = doc.get('metadata', {}).get('citation', '')
            
            # Skip if not a case
            if not citation or 'legislation' in doc.get('metadata', {}).get('type', ''):
                continue
            
            # Determine outcome
            outcome = None
            for outcome_type, patterns in outcome_patterns.items():
                if any(re.search(pattern, text) for pattern in patterns):
                    outcome = outcome_type
                    break
            
            if outcome:
                # Extract additional details
                case_data = {
                    'citation': citation,
                    'outcome': outcome,
                    'text_sample': text[:1000],
                    'factors': self._extract_case_factors(text)
                }
                
                # Extract compensation if mentioned
                comp_match = re.search(r'\$?([\d,]+)\s*(?:in\s+)?compensation', text)
                if comp_match:
                    case_data['compensation'] = int(comp_match.group(1).replace(',', ''))
                
                outcomes.append(case_data)
        
        return outcomes
    
    def _learn_winning_patterns(self, case_outcomes: List[Dict]) -> Dict:
        """Learn what patterns lead to wins"""
        
        winning_cases = [c for c in case_outcomes if c['outcome'] == 'applicant_won']
        losing_cases = [c for c in case_outcomes if c['outcome'] == 'applicant_lost']
        
        # Extract factor frequencies
        winning_factors = Counter()
        losing_factors = Counter()
        
        for case in winning_cases:
            for factor in case['factors']:
                winning_factors[factor] += 1
        
        for case in losing_cases:
            for factor in case['factors']:
                losing_factors[factor] += 1
        
        # Calculate win impact scores
        pattern_scores = {}
        
        for factor in set(winning_factors.keys()) | set(losing_factors.keys()):
            win_count = winning_factors.get(factor, 0)
            lose_count = losing_factors.get(factor, 0)
            total = win_count + lose_count
            
            if total > 5:  # Minimum sample size
                win_rate = win_count / total
                impact = win_rate - 0.5  # How much better than average
                
                pattern_scores[factor] = {
                    'win_rate': win_rate,
                    'impact': impact,
                    'occurrences': total,
                    'classification': 'strong_positive' if impact > 0.2 else 'positive' if impact > 0 else 'negative'
                }
        
        return pattern_scores
    
    def _extract_case_factors(self, text: str) -> List[str]:
        """Extract legal factors from case text"""
        factors = []
        
        factor_patterns = {
            'no_warning': r'no\s+(?:prior\s+)?warning',
            'long_service': r'\d+\s+years?\s+(?:of\s+)?(?:service|employment)',
            'summary_dismissal': r'summary\s+dismissal',
            'serious_misconduct': r'serious\s+misconduct',
            'procedural_fairness': r'procedural\s+fairness',
            'valid_reason': r'valid\s+reason',
            'harsh_unjust': r'harsh.*unjust.*unreasonable',
            'small_business': r'small\s+business',
            'genuine_redundancy': r'genuine\s+redundancy',
            'discrimination': r'discriminat\w+',
            'performance_management': r'performance\s+manage\w+'
        }
        
        for factor_name, pattern in factor_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                factors.append(factor_name)
        
        return factors
    
    def _extract_settlement_amounts(self, documents: List[Dict]) -> Dict:
        """Extract settlement patterns"""
        settlements = []
        
        settlement_patterns = [
            r'settled\s+for\s+\$?([\d,]+)',
            r'compensation\s+of\s+\$?([\d,]+)',
            r'payment\s+of\s+\$?([\d,]+)',
            r'(\d+)\s+weeks?\s+(?:of\s+)?pay'
        ]
        
        for doc in documents[:2000]:  # Sample for speed
            text = doc.get('text', '')
            
            for pattern in settlement_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        if 'week' in pattern:
                            # Convert weeks to approximate dollar amount
                            weeks = int(match)
                            amount = weeks * 1500  # Assume average weekly pay
                        else:
                            amount = int(match.replace(',', ''))
                        
                        if 1000 < amount < 500000:  # Reasonable range
                            settlements.append(amount)
                    except:
                        continue
        
        if settlements:
            return {
                'count': len(settlements),
                'average': np.mean(settlements),
                'median': np.median(settlements),
                'percentiles': {
                    '25th': np.percentile(settlements, 25),
                    '50th': np.percentile(settlements, 50),
                    '75th': np.percentile(settlements, 75),
                    '90th': np.percentile(settlements, 90)
                },
                'distribution': self._categorize_settlements(settlements)
            }
        
        return {}
    
    def _categorize_settlements(self, amounts: List[float]) -> Dict:
        """Categorize settlement amounts"""
        categories = {
            'small': len([a for a in amounts if a < 10000]),
            'medium': len([a for a in amounts if 10000 <= a < 50000]),
            'large': len([a for a in amounts if 50000 <= a < 100000]),
            'very_large': len([a for a in amounts if a >= 100000])
        }
        return categories
    
    def _build_precedent_network(self, documents: List[Dict]) -> List[Dict]:
        """Build network of precedents"""
        precedent_network = []
        
        # Extract citations between cases
        for doc in documents[:1000]:  # Sample
            text = doc.get('text', '')
            citing_case = doc.get('metadata', {}).get('citation', '')
            
            if not citing_case:
                continue
            
            # Find cited cases
            citation_pattern = r'(?:[A-Z]\w+\s+v\s+[A-Z]\w+.*?\[\d{4}\].*?(?:HCA|FCA|FWC|FCCA)\s*\d+)'
            cited_cases = re.findall(citation_pattern, text)
            
            if cited_cases:
                for cited in cited_cases[:5]:  # Limit per case
                    if cited != citing_case:
                        precedent_network.append({
                            'citing': citing_case,
                            'cited': cited,
                            'strength': 'strong' if cited in text[:2000] else 'moderate'
                        })
        
        return precedent_network
    
    def _analyze_judge_patterns(self, documents: List[Dict]) -> Dict:
        """Analyze patterns by judge"""
        judge_data = defaultdict(lambda: {'cases': 0, 'applicant_wins': 0, 'patterns': []})
        
        judge_pattern = r'(?:JUDGE|J\.|JUSTICE|COMMISSIONER|C\.|DEPUTY PRESIDENT)\s+([A-Z][A-Za-z]+)'
        
        for doc in documents[:1000]:
            text = doc.get('text', '')
            
            # Find judge
            judge_match = re.search(judge_pattern, text)
            if judge_match:
                judge_name = judge_match.group(1)
                
                # Determine outcome
                if re.search(r'application\s+granted|favor\s+of\s+applicant', text, re.IGNORECASE):
                    judge_data[judge_name]['applicant_wins'] += 1
                
                judge_data[judge_name]['cases'] += 1
        
        # Calculate win rates
        judge_analysis = {}
        for judge, data in judge_data.items():
            if data['cases'] >= 5:  # Minimum sample
                judge_analysis[judge] = {
                    'cases': data['cases'],
                    'applicant_win_rate': data['applicant_wins'] / data['cases'],
                    'tendency': 'applicant_friendly' if data['applicant_wins'] / data['cases'] > 0.6 else 'employer_friendly' if data['applicant_wins'] / data['cases'] < 0.4 else 'balanced'
                }
        
        return judge_analysis
    
    def _learn_temporal_patterns(self, documents: List[Dict]) -> Dict:
        """Learn patterns over time"""
        yearly_patterns = defaultdict(lambda: {'total': 0, 'wins': 0})
        
        year_pattern = r'(?:19|20)\d{2}'
        
        for doc in documents[:2000]:
            text = doc.get('text', '')[:500]  # Check beginning for year
            
            year_match = re.search(year_pattern, text)
            if year_match:
                year = int(year_match.group())
                
                if 2000 <= year <= 2024:
                    yearly_patterns[year]['total'] += 1
                    
                    if re.search(r'application\s+granted|favor\s+of\s+applicant', text, re.IGNORECASE):
                        yearly_patterns[year]['wins'] += 1
        
        # Calculate trends
        trends = {}
        for year, data in sorted(yearly_patterns.items()):
            if data['total'] >= 10:
                trends[year] = {
                    'cases': data['total'],
                    'win_rate': data['wins'] / data['total']
                }
        
        return trends
    
    def _create_case_clusters(self, documents: List[Dict]) -> Dict:
        """Create embedding clusters of similar cases"""
        
        # Get case texts
        case_texts = []
        case_metadata = []
        
        for doc in documents:
            if doc.get('metadata', {}).get('type') == 'decision':
                case_texts.append(doc['text'][:1000])
                case_metadata.append({
                    'citation': doc.get('metadata', {}).get('citation', 'Unknown'),
                    'outcome': 'unknown'
                })
        
        if len(case_texts) < 10:
            return {}
        
        print("Creating embeddings for clustering...")
        
        # Create embeddings
        embeddings = self.model.encode(case_texts[:200], show_progress_bar=True)  # Limit for speed
        
        # Simple clustering using cosine similarity
        from sklearn.cluster import KMeans
        
        n_clusters = min(10, len(embeddings) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_docs = [case_metadata[j] for j in range(len(clusters)) if clusters[j] == i]
            cluster_analysis[f'cluster_{i}'] = {
                'size': len(cluster_docs),
                'sample_cases': cluster_docs[:3]
            }
        
        return cluster_analysis
    
    def _generate_insights(self, case_outcomes: List[Dict], winning_patterns: Dict) -> Dict:
        """Generate actionable insights from learned patterns"""
        
        insights = {
            'success_factors': [],
            'risk_factors': [],
            'strategic_recommendations': []
        }
        
        # Top success factors
        for factor, data in sorted(winning_patterns.items(), key=lambda x: x[1].get('impact', 0), reverse=True)[:5]:
            if data.get('impact', 0) > 0.1:
                insights['success_factors'].append({
                    'factor': factor,
                    'win_rate': f"{data['win_rate']*100:.1f}%",
                    'impact': f"+{data['impact']*100:.1f}% above average"
                })
        
        # Top risk factors
        for factor, data in sorted(winning_patterns.items(), key=lambda x: x[1].get('impact', 0))[:5]:
            if data.get('impact', 0) < -0.1:
                insights['risk_factors'].append({
                    'factor': factor,
                    'win_rate': f"{data['win_rate']*100:.1f}%",
                    'impact': f"{data['impact']*100:.1f}% below average"
                })
        
        # Strategic recommendations
        if insights['success_factors']:
            insights['strategic_recommendations'].append(
                f"Focus on establishing: {', '.join([f['factor'] for f in insights['success_factors'][:3]])}"
            )
        
        if insights['risk_factors']:
            insights['strategic_recommendations'].append(
                f"Address or mitigate: {', '.join([f['factor'] for f in insights['risk_factors'][:3]])}"
            )
        
        return insights

# Save learned intelligence
def save_corpus_intelligence(intelligence: Dict, filename: str = 'corpus_intelligence.json'):
    """Save extracted intelligence"""
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    clean_intelligence = convert_numpy(intelligence)
    
    with open(filename, 'w') as f:
        json.dump(clean_intelligence, f, indent=2, default=str)
    
    print(f"✅ Saved intelligence to {filename}")

if __name__ == "__main__":
    print("🧠 CORPUS INTELLIGENCE EXTRACTOR")
    print("=" * 60)
    
    # Load corpus
    print("Loading corpus...")
    with open('data/simple_index.pkl', 'rb') as f:
        data = pickle.load(f)
        documents = data['documents']
    
    print(f"Loaded {len(documents)} documents")
    
    # Extract intelligence
    extractor = CorpusIntelligenceExtractor()
    intelligence = extractor.extract_intelligence(documents)
    
    # Save results
    save_corpus_intelligence(intelligence)
    
    # Print summary
    print("\n📊 INTELLIGENCE SUMMARY")
    print("=" * 60)
    print(f"✅ Analyzed {len(intelligence['case_outcomes'])} case outcomes")
    print(f"✅ Discovered {len(intelligence['winning_patterns'])} legal patterns")
    
    if intelligence['settlement_intelligence']:
        print(f"✅ Average settlement: ${intelligence['settlement_intelligence']['average']:,.0f}")
        print(f"✅ Median settlement: ${intelligence['settlement_intelligence']['median']:,.0f}")
    
    if intelligence['corpus_insights']['success_factors']:
        print("\n🎯 TOP SUCCESS FACTORS:")
        for factor in intelligence['corpus_insights']['success_factors'][:3]:
            print(f"   - {factor['factor']}: {factor['win_rate']} win rate ({factor['impact']})")
    
    print("\n💾 Intelligence saved to corpus_intelligence.json")
