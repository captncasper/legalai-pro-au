"""
RAG Integration Module for Australian Legal AI
Adds vector search capabilities to the main application
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import faiss
import pickle

logger = logging.getLogger(__name__)

@dataclass
class RAGDocument:
    """Represents a document in the RAG index"""
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
class LegalRAGIndexer:
    """Lightweight RAG indexer using FAISS for vector search"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.doc_map = {}
        
    def initialize_index(self):
        """Initialize FAISS index"""
        # Use IndexFlatL2 for simplicity - can upgrade to IVF for larger datasets
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info("🔍 Initialized FAISS index for RAG")
        
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """Add documents with their embeddings to the index"""
        if self.index is None:
            self.initialize_index()
            
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store document mapping
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            doc_id = f"doc_{start_idx + i}"
            self.documents.append(doc)
            self.doc_map[doc_id] = start_idx + i
            
        logger.info(f"📚 Added {len(documents)} documents to RAG index")
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("RAG index is empty")
            return []
            
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'document': doc,
                    'score': float(1 / (1 + distances[0][i])),  # Convert distance to similarity
                    'distance': float(distances[0][i])
                })
                
        return results
        
    def save_index(self, path: str):
        """Save the index to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        if self.index and self.index.ntotal > 0:
            faiss.write_index(self.index, f"{path}.faiss")
            
        # Save documents and metadata
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_map': self.doc_map,
                'embedding_dim': self.embedding_dim
            }, f)
            
        logger.info(f"💾 Saved RAG index to {path}")
        
    def load_index(self, path: str) -> bool:
        """Load index from disk"""
        try:
            # Load FAISS index
            if os.path.exists(f"{path}.faiss"):
                self.index = faiss.read_index(f"{path}.faiss")
                
            # Load documents and metadata
            if os.path.exists(f"{path}.pkl"):
                with open(f"{path}.pkl", 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.doc_map = data['doc_map']
                    self.embedding_dim = data['embedding_dim']
                    
                logger.info(f"📂 Loaded RAG index with {len(self.documents)} documents")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load RAG index: {e}")
            
        return False

def create_legal_rag_index(documents: List[Dict], model) -> LegalRAGIndexer:
    """Create a RAG index from legal documents"""
    indexer = LegalRAGIndexer()
    
    # Process in batches
    batch_size = 50
    all_embeddings = []
    
    logger.info("🔨 Building RAG index...")
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        # Extract text for embedding
        texts = []
        for doc in batch:
            text = doc.get('text', '')
            # Truncate long texts
            if len(text) > 1000:
                text = text[:1000] + "..."
            texts.append(text)
            
        # Generate embeddings
        if model and hasattr(model, 'encode'):
            embeddings = model.encode(texts, show_progress_bar=False)
            all_embeddings.append(embeddings)
    
    if all_embeddings:
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        # Add to index
        indexer.add_documents(documents, all_embeddings)
        
    return indexer

# Integration functions for main app
def enhance_search_with_rag(query: str, rag_indexer: LegalRAGIndexer, model, k: int = 5) -> List[Dict]:
    """Enhance search results using RAG"""
    if not rag_indexer or not model:
        return []
        
    try:
        # Generate query embedding
        query_embedding = model.encode([query], show_progress_bar=False)[0]
        
        # Search in RAG index
        results = rag_indexer.search(query_embedding, k=k)
        
        # Format results
        enhanced_results = []
        for result in results:
            doc = result['document']
            enhanced_results.append({
                'title': doc.get('title', 'Legal Document'),
                'text': doc.get('text', '')[:500],
                'score': result['score'],
                'type': doc.get('type', 'document'),
                'metadata': doc.get('metadata', {}),
                'rag_score': result['score']
            })
            
        return enhanced_results
        
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return []

def add_rag_to_app(app_instance, legal_corpus: List[Dict], semantic_model):
    """Add RAG functionality to existing app"""
    logger.info("🚀 Initializing RAG system...")
    
    # Create RAG indexer
    rag_indexer = LegalRAGIndexer()
    
    # Check if index exists
    index_path = "./rag_index/legal_rag"
    if os.path.exists(f"{index_path}.faiss"):
        # Load existing index
        if rag_indexer.load_index(index_path):
            logger.info("✅ Loaded existing RAG index")
        else:
            # Build new index
            if semantic_model and legal_corpus:
                rag_indexer = create_legal_rag_index(legal_corpus, semantic_model)
                rag_indexer.save_index(index_path)
    else:
        # Build new index
        if semantic_model and legal_corpus:
            rag_indexer = create_legal_rag_index(legal_corpus, semantic_model)
            rag_indexer.save_index(index_path)
            
    return rag_indexer