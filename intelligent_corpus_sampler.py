#!/usr/bin/env python3
"""
Intelligent Corpus Sampler
Selectively loads the BEST documents from 260k corpus
"""

import pickle
import json
import re
from typing import List, Dict, Set
import random
from collections import defaultdict
import hashlib
import gzip
import numpy as np

class IntelligentCorpusSampler:
    def __init__(self):
        self.priority_patterns = {
            'high_value_cases': {
                'patterns': [
                    r'high\s*court',
                    r'federal\s*court',
                    r'full\s*bench',
                    r'appeal.*allow',
                    r'landmark',
                    r'precedent'
                ],
                'weight': 10
            },
            'key_legislation': {
                'patterns': [
                    r'fair\s*work\s*act',
                    r'discrimination\s*act',
                    r'workplace\s*relations',
                    r'section\s*\d+',
                ],
                'weight': 8
            },
            'recent_cases': {
                'patterns': [
                    r'202[0-4]',
                    r'2019'
                ],
                'weight': 7
            },
            'compensation_cases': {
                'patterns': [
                    r'\$\d{4,}',
                    r'compensation.*order',
                    r'weeks.*pay',
                    r'settlement'
                ],
                'weight': 6
            },
            'common_scenarios': {
                'patterns': [
                    r'unfair.*dismissal',
                    r'discrimination',
                    r'bullying',
                    r'redundancy',
                    r'breach.*contract'
                ],
                'weight': 5
            }
        }
    
    def smart_sample(self, all_documents: List[Dict], target_size: int = 50000) -> Dict:
        """Intelligently sample the most valuable documents"""
        
        print(f"🧠 Intelligently sampling {target_size} from {len(all_documents)} documents...")
        
        # Score all documents
        doc_scores = []
        doc_categories = defaultdict(list)
        
        for i, doc in enumerate(all_documents):
            if i % 10000 == 0:
                print(f"Scoring... {i}/{len(all_documents)}")
            
            score, categories = self._score_document(doc)
            doc_scores.append((i, score))
            
            for cat in categories:
                doc_categories[cat].append(i)
        
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select documents with diversity
        selected_indices = set()
        
        # 1. Take top scored documents (40%)
        top_count = int(target_size * 0.4)
        for idx, score in doc_scores[:top_count]:
            selected_indices.add(idx)
        
        # 2. Ensure category representation (30%)
        category_count = int(target_size * 0.3 / len(doc_categories))
        for category, indices in doc_categories.items():
            # Take best from each category
            category_docs = [(idx, doc_scores[idx][1]) for idx in indices if idx not in selected_indices]
            category_docs.sort(key=lambda x: x[1], reverse=True)
            
            for idx, _ in category_docs[:category_count]:
                selected_indices.add(idx)
                if len(selected_indices) >= target_size * 0.7:
                    break
        
        # 3. Add random sample for diversity (30%)
        remaining = target_size - len(selected_indices)
        unselected = [i for i in range(len(all_documents)) if i not in selected_indices]
        random_sample = random.sample(unselected, min(remaining, len(unselected)))
        selected_indices.update(random_sample)
        
        # Build final corpus
        sampled_documents = [all_documents[i] for i in sorted(selected_indices)]
        
        # Create metadata about sampling
        sampling_metadata = {
            'original_size': len(all_documents),
            'sampled_size': len(sampled_documents),
            'sampling_method': 'intelligent_scoring',
            'category_distribution': {cat: len(indices) for cat, indices in doc_categories.items()},
            'score_distribution': {
                'top_10_percent': sum(1 for _, s in doc_scores if s > np.percentile([s for _, s in doc_scores], 90)),
                'top_25_percent': sum(1 for _, s in doc_scores if s > np.percentile([s for _, s in doc_scores], 75)),
            }
        }
        
        return {
            'documents': sampled_documents,
            'metadata': sampling_metadata,
            'indices': list(selected_indices)
        }
    
    def _score_document(self, doc: Dict) -> tuple:
        """Score document importance"""
        text = doc.get('text', '').lower()
        metadata = doc.get('metadata', {})
        
        score = 0
        categories = []
        
        # Check priority patterns
        for category, config in self.priority_patterns.items():
            matches = sum(1 for pattern in config['patterns'] if re.search(pattern, text))
            if matches > 0:
                score += matches * config['weight']
                categories.append(category)
        
        # Bonus for citations
        if metadata.get('citation'):
            score += 5
            
            # Extra bonus for important courts
            if any(court in metadata['citation'] for court in ['HCA', 'FCA', 'FCAFC']):
                score += 10
        
        # Bonus for recent content
        year_match = re.search(r'20(\d{2})', text[:500])
        if year_match:
            year = int(year_match.group(1))
            if year >= 20:  # 2020+
                score += 5
        
        # Length penalty (avoid too short or too long)
        text_len = len(text)
        if 1000 < text_len < 50000:
            score += 3
        elif text_len > 100000:
            score -= 5  # Too long, probably concatenated
        
        return score, categories

class CompressedCorpusBuilder:
    """Build compressed, searchable corpus"""
    
    @staticmethod
    def build_compressed_index(documents: List[Dict], output_file: str = 'compressed_corpus.pkl.gz'):
        """Build compressed corpus with smart indexing"""
        
        print("📦 Building compressed corpus...")
        
        # Build multiple indexes
        corpus = {
            'documents': documents,
            'indexes': {
                'keyword_index': CompressedCorpusBuilder._build_keyword_index(documents),
                'citation_index': CompressedCorpusBuilder._build_citation_index(documents),
                'type_index': CompressedCorpusBuilder._build_type_index(documents),
                'year_index': CompressedCorpusBuilder._build_year_index(documents),
                'outcome_index': CompressedCorpusBuilder._build_outcome_index(documents)
            },
            'statistics': {
                'total_documents': len(documents),
                'unique_citations': len(set(d.get('metadata', {}).get('citation', '') for d in documents)),
                'document_types': dict(CompressedCorpusBuilder._count_types(documents))
            }
        }
        
        # Compress and save
        print("💾 Compressing and saving...")
        with gzip.open(output_file, 'wb') as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Check size
        import os
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Compressed corpus saved: {size_mb:.1f} MB")
        
        return corpus
    
    @staticmethod
    def _build_keyword_index(documents: List[Dict]) -> Dict:
        """Build keyword index"""
        keyword_index = defaultdict(set)
        
        important_keywords = [
            'dismissal', 'unfair', 'discrimination', 'harassment', 'bullying',
            'redundancy', 'compensation', 'reinstatement', 'breach', 'contract',
            'wages', 'overtime', 'safety', 'injury', 'stress'
        ]
        
        for i, doc in enumerate(documents):
            text = doc.get('text', '').lower()
            for keyword in important_keywords:
                if keyword in text:
                    keyword_index[keyword].add(i)
        
        # Convert sets to lists for pickling
        return {k: list(v) for k, v in keyword_index.items()}
    
    @staticmethod
    def _build_citation_index(documents: List[Dict]) -> Dict:
        """Build citation lookup"""
        return {
            doc.get('metadata', {}).get('citation', ''): i 
            for i, doc in enumerate(documents) 
            if doc.get('metadata', {}).get('citation')
        }
    
    @staticmethod
    def _build_type_index(documents: List[Dict]) -> Dict:
        """Index by document type"""
        type_index = defaultdict(list)
        for i, doc in enumerate(documents):
            doc_type = doc.get('metadata', {}).get('type', 'unknown')
            type_index[doc_type].append(i)
        return dict(type_index)
    
    @staticmethod
    def _build_year_index(documents: List[Dict]) -> Dict:
        """Index by year"""
        year_index = defaultdict(list)
        
        for i, doc in enumerate(documents):
            text = doc.get('text', '')[:500]
            year_match = re.search(r'(19|20)\d{2}', text)
            if year_match:
                year = int(year_match.group())
                if 1990 <= year <= 2024:
                    year_index[year].append(i)
        
        return dict(year_index)
    
    @staticmethod
    def _build_outcome_index(documents: List[Dict]) -> Dict:
        """Index by case outcome"""
        outcome_index = defaultdict(list)
        
        outcome_patterns = {
            'applicant_won': ['application granted', 'found in favor of applicant'],
            'applicant_lost': ['application dismissed', 'found against applicant'],
            'settled': ['matter settled', 'consent order']
        }
        
        for i, doc in enumerate(documents):
            text = doc.get('text', '').lower()
            for outcome, patterns in outcome_patterns.items():
                if any(pattern in text for pattern in patterns):
                    outcome_index[outcome].append(i)
        
        return dict(outcome_index)
    
    @staticmethod
    def _count_types(documents: List[Dict]) -> Dict:
        """Count document types"""
        from collections import Counter
        types = [doc.get('metadata', {}).get('type', 'unknown') for doc in documents]
        return Counter(types)

# Memory-efficient corpus loader
class StreamingCorpusProcessor:
    """Process large corpus in streams to save memory"""
    
    @staticmethod
    def process_in_chunks(file_path: str, chunk_size: int = 10000):
        """Process corpus in chunks"""
        
        # This would connect to your full corpus
        # For now, showing the structure
        
        chunk_processor = {
            'extract_patterns': lambda chunk: StreamingCorpusProcessor._extract_patterns(chunk),
            'extract_outcomes': lambda chunk: StreamingCorpusProcessor._extract_outcomes(chunk),
            'extract_settlements': lambda chunk: StreamingCorpusProcessor._extract_settlements(chunk)
        }
        
        aggregated_results = defaultdict(list)
        
        # Process each chunk
        # In real implementation, this would read from your 260k corpus
        print(f"Processing corpus in chunks of {chunk_size}...")
        
        return aggregated_results
    
    @staticmethod
    def _extract_patterns(chunk: List[Dict]) -> Dict:
        """Extract patterns from chunk"""
        patterns = defaultdict(int)
        
        pattern_list = [
            'no warning', 'long service', 'discrimination',
            'performance management', 'serious misconduct'
        ]
        
        for doc in chunk:
            text = doc.get('text', '').lower()
            for pattern in pattern_list:
                if pattern in text:
                    patterns[pattern] += 1
        
        return dict(patterns)
    
    @staticmethod
    def _extract_outcomes(chunk: List[Dict]) -> Dict:
        """Extract outcomes from chunk"""
        outcomes = {'won': 0, 'lost': 0, 'settled': 0}
        
        for doc in chunk:
            text = doc.get('text', '').lower()
            if 'application granted' in text:
                outcomes['won'] += 1
            elif 'application dismissed' in text:
                outcomes['lost'] += 1
            elif 'settled' in text:
                outcomes['settled'] += 1
        
        return outcomes
    
    @staticmethod
    def _extract_settlements(chunk: List[Dict]) -> List[int]:
        """Extract settlement amounts"""
        amounts = []
        
        for doc in chunk:
            text = doc.get('text', '')
            amount_matches = re.findall(r'\$(\d{1,3}(?:,\d{3})*)', text)
            for match in amount_matches:
                try:
                    amount = int(match.replace(',', ''))
                    if 1000 < amount < 500000:  # Reasonable range
                        amounts.append(amount)
                except:
                    continue
        
        return amounts

if __name__ == "__main__":
    print("🧠 INTELLIGENT CORPUS MANAGEMENT SYSTEM")
    print("=" * 60)
    
    # For demo, use existing corpus
    print("Loading current corpus...")
    with open('data/simple_index.pkl', 'rb') as f:
        data = pickle.load(f)
        current_documents = data['documents']
    
    print(f"Current corpus: {len(current_documents)} documents")
    
    # Option 1: Intelligent sampling
    print("\n📊 OPTION 1: Intelligent Sampling")
    print("This would sample the BEST 50k docs from your 260k corpus")
    sampler = IntelligentCorpusSampler()
    
    # Demo with current corpus
    sampled_data = sampler.smart_sample(current_documents, target_size=min(5000, len(current_documents)))
    
    print(f"\nSampling results:")
    print(f"- Selected {sampled_data['metadata']['sampled_size']} documents")
    print(f"- Categories found: {list(sampled_data['metadata']['category_distribution'].keys())}")
    
    # Option 2: Compressed corpus
    print("\n📦 OPTION 2: Compressed Indexed Corpus")
    builder = CompressedCorpusBuilder()
    
    # Build compressed version (demo with subset)
    compressed_corpus = builder.build_compressed_index(
        sampled_data['documents'][:1000],  # Demo subset
        'demo_compressed_corpus.pkl.gz'
    )
    
    print(f"\nIndexes created:")
    for index_name, index_data in compressed_corpus['indexes'].items():
        print(f"- {index_name}: {len(index_data)} entries")
    
    # Option 3: Streaming processor
    print("\n🌊 OPTION 3: Streaming Processing")
    print("This would process your 260k corpus in memory-efficient chunks")
    print("Perfect for Codespaces with limited memory")
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. Use intelligent sampling to get best 50k docs")
    print("2. Build compressed indexes for fast search")
    print("3. Use streaming for pattern extraction")
    print("4. Store learned patterns, not all documents")
