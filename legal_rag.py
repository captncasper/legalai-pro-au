import numpy as np
#!/usr/bin/env python3
"""
Legal RAG Query Engine - No hallucinations, only facts!
"""

from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict

class LegalRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="./rag_index")
        self.collection = self.client.get_collection("aussie_legal")
        
    def query(self, question: str, n_results: int = 5) -> Dict:
        """Query with semantic search and return citations"""
        
        # Embed question
        query_embedding = self.model.encode(question)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        # Format with citations
        formatted_results = []
        seen_citations = set()
        
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if 'distances' in results else 0
            
            citation = metadata['citation']
            if citation not in seen_citations:
                seen_citations.add(citation)
                formatted_results.append({
                    'text': doc,
                    'citation': citation,
                    'confidence': 1 - (distance/2),
                    'type': metadata.get('type', 'unknown')
                })
        
        return {
            'question': question,
            'sources': formatted_results,
            'answer': self._generate_answer(question, formatted_results)
        }
    
    def _generate_answer(self, question: str, sources: List[Dict]) -> str:
        """Generate answer from sources - no hallucination!"""
        if not sources:
            return "No relevant legal documents found."
        
        # Build answer from top sources
        answer = "Based on Australian legal documents:\n\n"
        
        for i, source in enumerate(sources[:3], 1):
            answer += f"{i}. {source['citation']}:\n"
            answer += f"   {source['text'][:200]}...\n\n"
        
        return answer

# Test it
if __name__ == "__main__":
    rag = LegalRAG()
    
    # Test queries
    queries = [
        "What are the time limits for unfair dismissal?",
        "Can a contractor claim unfair dismissal?",
        "What constitutes serious misconduct?"
    ]
    
    for q in queries:
        print(f"\n❓ Question: {q}")
        result = rag.query(q)
        print(f"📚 Sources: {len(result['sources'])} documents found")
        print(f"✅ Answer: {result['answer'][:300]}...")
