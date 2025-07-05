#!/usr/bin/env python3
"""Add semantic search to your corpus"""

import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from load_real_aussie_corpus import corpus

class SemanticSearchEngine:
    def __init__(self):
        print("Loading semantic model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus_embeddings = None
        
    def create_embeddings(self):
        """Create embeddings for all cases"""
        corpus.load_corpus()
        print(f"Creating embeddings for {len(corpus.cases)} cases...")
        
        # Create searchable text for each case
        texts = []
        for case in corpus.cases:
            searchable_text = f"{case['case_name']} {case['text']} {case['outcome']}"
            texts.append(searchable_text)
        
        # Generate embeddings
        self.corpus_embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Save embeddings
        with open('case_embeddings.pkl', 'wb') as f:
            pickle.dump({
                'embeddings': self.corpus_embeddings,
                'case_citations': [c['citation'] for c in corpus.cases]
            }, f)
        
        print(f"✅ Saved embeddings for {len(self.corpus_embeddings)} cases")
    
    def semantic_search(self, query: str, top_k: int = 10):
        """Search using semantic similarity"""
        if self.corpus_embeddings is None:
            # Load saved embeddings
            with open('case_embeddings.pkl', 'rb') as f:
                data = pickle.load(f)
                self.corpus_embeddings = data['embeddings']
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(self.corpus_embeddings, query_embedding.T).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            case = corpus.cases[idx]
            results.append({
                'case': case,
                'similarity_score': float(similarities[idx])
            })
        
        return results

# Run this to create embeddings
if __name__ == "__main__":
    engine = SemanticSearchEngine()
    engine.create_embeddings()
    
    # Test semantic search
    print("\n🔍 Testing Semantic Search:")
    results = engine.semantic_search("negligence personal injury damages")
    print(f"Found {len(results)} semantically similar cases")
    for r in results[:3]:
        print(f"  - {r['case']['citation']} (score: {r['similarity_score']:.3f})")
