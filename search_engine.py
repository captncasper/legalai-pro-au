import numpy as np
import pickle
import re
from collections import Counter

class SimpleSearchEngine:
    def __init__(self, index_path='data/simple_index.pkl'):
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.keyword_index = data['keyword_index']
        print(f"Loaded search index with {len(self.documents)} documents")
    
    def search(self, query, num_results=5):
        # Clean and split query
        query_words = re.findall(r'\w+', query.lower())
        
        # Score documents by keyword matches
        doc_scores = Counter()
        for word in query_words:
            if word in self.keyword_index:
                for doc_id in self.keyword_index[word]:
                    doc_scores[doc_id] += 1
        
        # Get top results
        results = []
        for doc_id, score in doc_scores.most_common(num_results):
            doc = self.documents[doc_id]
            results.append({
                'text': doc['text'],
                'metadata': doc.get('metadata', {}),
                'score': score,
                'citation': doc.get('metadata', {}).get('citation', 'Unknown')
            })
        
        return results

# Initialize the search engine
search_engine = SimpleSearchEngine()