"""Build search index from Australian Legal Corpus JSONL file"""
import os
import sys
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.embeddings import LegalEmbedder


class AustralianLegalIndexer:
    def __init__(self, jsonl_path: str):
        self.jsonl_path = jsonl_path
        self.embedder = LegalEmbedder()
        self.documents = []
        self.metadata = []
        
    def load_corpus(self) -> int:
        """Load documents from JSONL file"""
        print(f"📚 Loading corpus from {self.jsonl_path}")
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading documents"):
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    meta = data.get('metadata', {})
                    
                    # Clean and validate text
                    if len(text) > 100:  # Skip very short documents
                        self.documents.append(text)
                        self.metadata.append(meta)
                except Exception as e:
                    print(f"Error loading line: {e}")
                    continue
        
        print(f"✅ Loaded {len(self.documents)} documents")
        return len(self.documents)
    
    def analyze_corpus(self):
        """Analyze the loaded corpus"""
        print("\n📊 Corpus Analysis:")
        
        # Jurisdiction distribution
        jurisdictions = defaultdict(int)
        doc_types = defaultdict(int)
        sources = defaultdict(int)
        
        for meta in self.metadata:
            jurisdictions[meta.get('jurisdiction', 'Unknown')] += 1
            doc_types[meta.get('type', 'Unknown')] += 1
            sources[meta.get('source', 'Unknown')] += 1
        
        print(f"\n🏛️ Jurisdictions ({len(jurisdictions)}):")
        for jur, count in sorted(jurisdictions.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {jur}: {count} documents")
        
        print(f"\n📄 Document Types ({len(doc_types)}):")
        for dtype, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dtype}: {count} documents")
        
        print(f"\n📚 Sources ({len(sources)}):")
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {source}: {count} documents")
        
        # Text statistics
        text_lengths = [len(doc) for doc in self.documents]
        print(f"\n📏 Text Length Statistics:")
        print(f"  Average: {np.mean(text_lengths):.0f} characters")
        print(f"  Median: {np.median(text_lengths):.0f} characters")
        print(f"  Min: {np.min(text_lengths)} characters")
        print(f"  Max: {np.max(text_lengths)} characters")
    
    def build_index(self, output_dir: str = "data", batch_size: int = 32):
        """Build FAISS index with the real Australian legal documents"""
        
        if not self.documents:
            print("❌ No documents loaded!")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🧮 Creating embeddings for {len(self.documents)} documents...")
        print(f"   Batch size: {batch_size}")
        print(f"   Estimated time: {len(self.documents) / batch_size / 10:.1f} minutes")
        
        # Create embeddings in batches
        all_embeddings = []
        
        for i in tqdm(range(0, len(self.documents), batch_size), desc="Creating embeddings"):
            batch_docs = self.documents[i:i + batch_size]
            
            # Create embeddings
            embeddings = self.embedder.encode(batch_docs)
            all_embeddings.append(embeddings)
        
        # Combine all embeddings
        embeddings_matrix = np.vstack(all_embeddings)
        print(f"✅ Created embeddings matrix: {embeddings_matrix.shape}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        # Build FAISS index
        print("\n🔍 Building FAISS index...")
        dimension = embeddings_matrix.shape[1]
        
        # Use IndexFlatIP for exact search (inner product = cosine similarity after normalization)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_matrix)
        
        print(f"✅ Added {index.ntotal} vectors to index")
        
        # Save everything
        print("\n💾 Saving index and data...")
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "legal_index.faiss")
        faiss.write_index(index, index_path)
        print(f"  ✓ Index: {index_path}")
        
        # Save documents
        docs_path = os.path.join(output_dir, "legal_documents.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"  ✓ Documents: {docs_path}")
        
        # Save metadata
        meta_path = os.path.join(output_dir, "legal_metadata.pkl")
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"  ✓ Metadata: {meta_path}")
        
        # Save index statistics
        stats = {
            'total_documents': len(self.documents),
            'embedding_dimension': dimension,
            'index_type': 'IndexFlatIP',
            'jurisdictions': len(set(m.get('jurisdiction', '') for m in self.metadata)),
            'document_types': len(set(m.get('type', '') for m in self.metadata)),
            'average_doc_length': np.mean([len(doc) for doc in self.documents]),
            'created_date': str(np.datetime64('today'))
        }
        
        stats_path = os.path.join(output_dir, "index_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  ✓ Statistics: {stats_path}")
        
        print(f"\n🎉 Index building complete!")
        print(f"📊 Total size: {self._get_dir_size(output_dir):.2f} MB")
        
        return stats
    
    def _get_dir_size(self, path: str) -> float:
        """Get directory size in MB"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size / (1024 * 1024)
    
    def test_search(self, query: str = "unfair dismissal Fair Work Act", k: int = 5):
        """Test the search functionality"""
        print(f"\n🔍 Testing search with query: '{query}'")
        
        # Create query embedding
        query_embedding = self.embedder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Load index
        index = faiss.read_index("data/legal_index.faiss")
        
        # Search
        scores, indices = index.search(query_embedding, k)
        
        print(f"\n📊 Top {k} results:")
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            doc = self.documents[idx]
            meta = self.metadata[idx]
            
            print(f"\n{i+1}. Score: {score:.3f}")
            print(f"   Type: {meta.get('type', 'Unknown')}")
            print(f"   Jurisdiction: {meta.get('jurisdiction', 'Unknown')}")
            print(f"   Citation: {meta.get('citation', 'N/A')}")
            print(f"   Preview: {doc[:200]}...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Index Australian Legal Corpus")
    parser.add_argument("--input", type=str, default="corpus_export/australian_legal_corpus.jsonl",
                       help="Path to JSONL corpus file")
    parser.add_argument("--output", type=str, default="data",
                       help="Output directory for index")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding creation")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze corpus statistics")
    parser.add_argument("--test", action="store_true",
                       help="Test search after building")
    
    args = parser.parse_args()
    
    # Create indexer
    indexer = AustralianLegalIndexer(args.input)
    
    # Load corpus
    num_docs = indexer.load_corpus()
    if num_docs == 0:
        print("❌ No documents found!")
        return
    
    # Analyze if requested
    if args.analyze:
        indexer.analyze_corpus()
    
    # Build index
    stats = indexer.build_index(args.output, args.batch_size)
    
    # Test if requested
    if args.test:
        indexer.test_search()
        
        # Try another query
        indexer.test_search("directors duties Corporations Act", k=3)
        
        # Try jurisdiction-specific
        indexer.test_search("native title Queensland", k=3)
    
    print("\n✅ Your Australian Legal AI search engine is ready!")
    print("🚀 Restart your API server to use the new index")


if __name__ == "__main__":
    main()