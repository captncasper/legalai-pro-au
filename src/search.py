"""Legal document search functionality with demo mode"""
import os
import sys
import pickle
import numpy as np
from typing import List, Dict, Any, Optional

# Try to import dependencies with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  FAISS not available, using simple search")
    FAISS_AVAILABLE = False

try:
    from .embeddings import LegalEmbedder
    EMBEDDER_AVAILABLE = True
except ImportError:
    print("⚠️  Embedder not available, using mock embeddings")
    EMBEDDER_AVAILABLE = False


class MockEmbedder:
    """Simple mock embedder for demo purposes"""
    def encode(self, texts: List[str]) -> np.ndarray:
        # Simple hash-based embedding for demo
        embeddings = []
        for text in texts:
            # Create a simple vector based on text features
            vec = np.random.RandomState(hash(text) % 2**32).rand(768).astype('float32')
            embeddings.append(vec)
        return np.array(embeddings)


class LegalSearchEngine:
    def __init__(self, 
                 index_path: str = "data/legal_index.faiss", 
                 docs_path: str = "data/legal_documents.pkl",
                 demo_mode: bool = False):
        """Initialize search engine with automatic demo fallback"""
        
        self.demo_mode = demo_mode
        self.index_path = index_path
        self.docs_path = docs_path
        
        # Initialize embedder
        if EMBEDDER_AVAILABLE and not demo_mode:
            try:
                self.embedder = LegalEmbedder()
                print("✅ Using real embedder")
            except Exception as e:
                print(f"⚠️  Embedder initialization failed: {e}")
                self.embedder = MockEmbedder()
                self.demo_mode = True
        else:
            self.embedder = MockEmbedder()
            self.demo_mode = True
        
        # Load or create index
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            print("📦 Creating demo index...")
            self._create_demo_index()
        else:
            self._load_index()
    
    def _load_index(self):
        """Load existing index and documents"""
        try:
            if FAISS_AVAILABLE:
                self.index = faiss.read_index(self.index_path)
            else:
                # Simple fallback - just store embeddings
                with open(self.index_path.replace('.faiss', '_embeddings.pkl'), 'rb') as f:
                    self.embeddings = pickle.load(f)
                self.index = None
            
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            print(f"✅ Loaded index with {len(self.documents)} documents")
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            self._create_demo_index()
    
    def _create_demo_index(self):
        """Create a comprehensive demo index for testing"""
        os.makedirs("data", exist_ok=True)
        
        # Comprehensive demo documents covering various areas of Australian law
        self.documents = [
            # Employment Law
            "The Fair Work Act 2009 (Cth) establishes the National Employment Standards (NES) which provide minimum entitlements for employees including maximum weekly hours, flexible working arrangements, parental leave, annual leave, personal/carer's leave, community service leave, long service leave, public holidays, notice of termination and redundancy pay.",
            
            "Under the Fair Work Act, unfair dismissal claims can be made by employees who have completed the minimum employment period (6 months for businesses with 15+ employees, 12 months for smaller businesses). The dismissal must be harsh, unjust or unreasonable, and procedural fairness must be considered.",
            
            # Contract Law
            "Australian contract law requires five essential elements for a valid contract: offer, acceptance, consideration, intention to create legal relations, and capacity. The objective test from Smith v Hughes applies - what would a reasonable person understand the agreement to be?",
            
            "The doctrine of frustration in Australian contract law, as established in Codelfa Construction Pty Ltd v State Rail Authority of NSW, applies when unforeseen circumstances make performance impossible or fundamentally different from what was agreed.",
            
            # Corporate Law
            "Directors' duties under the Corporations Act 2001 include: duty of care and diligence (s180), duty of good faith (s181), duty not to improperly use position (s182), and duty not to improperly use information (s183). Breach can result in civil penalties up to $1.11 million.",
            
            "The corporate veil in Australia can be pierced in limited circumstances including fraud, agency, or where the company is a mere facade. The High Court in Briggs v James Hardie & Co Pty Ltd confirmed the separate legal entity principle remains strong.",
            
            # Property Law
            "The Torrens system of land registration in Australia provides for registration of title, with the register serving as conclusive evidence of ownership. Indefeasibility of title is subject to exceptions including fraud, personal equities, and statutory exceptions.",
            
            "Native title rights and interests are recognized and protected under the Native Title Act 1993 (Cth). The Mabo decision overturned terra nullius and recognized that Indigenous Australians have rights to land based on traditional laws and customs.",
            
            # Criminal Law
            "Criminal responsibility in Australia requires both actus reus (guilty act) and mens rea (guilty mind). The Criminal Code Act 1995 (Cth) codifies federal criminal law, while states have their own criminal codes or rely on common law.",
            
            "The right to silence in Australia means an accused person cannot be compelled to give evidence at trial. However, in NSW, adverse inferences can be drawn from silence during police questioning in serious indictable offences under the Evidence Act modifications.",
            
            # Constitutional Law
            "The Australian Constitution establishes a federal system with a division of powers between Commonwealth and States. Section 51 enumerates Commonwealth legislative powers, while residual powers remain with the States under s107.",
            
            "The implied freedom of political communication, established in Australian Capital Television v Commonwealth, protects communication on political and government matters but is not an absolute right and must be balanced against legitimate purposes.",
            
            # Family Law
            "In parenting disputes under the Family Law Act 1975, the best interests of the child is the paramount consideration. Section 60CC sets out primary and additional considerations including the benefit of meaningful relationships with both parents.",
            
            "Property settlement in family law follows a four-step process: identify and value the property pool, assess contributions (financial and non-financial), consider future needs under s75(2), and determine a just and equitable division.",
            
            # Tort Law
            "Negligence in Australia requires duty of care, breach of that duty, causation, and damage. The Civil Liability Acts have modified common law, introducing proportionate liability and capping damages for personal injury.",
            
            "Defamation law was harmonized across Australia in 2005. A plaintiff must prove publication of defamatory material that identifies them. Defences include truth, honest opinion, fair report of proceedings of public concern, and qualified privilege.",
            
            # Consumer Law
            "The Australian Consumer Law provides consumer guarantees that cannot be excluded, including that goods are of acceptable quality, fit for purpose, and match description. Remedies include repair, replacement, or refund depending on whether the failure is major or minor.",
            
            "Misleading and deceptive conduct under s18 of the Australian Consumer Law prohibits conduct in trade or commerce that is likely to mislead or deceive. Intention is irrelevant - the test is whether the conduct is likely to mislead the target audience.",
            
            # Privacy Law
            "The Privacy Act 1988 (Cth) includes 13 Australian Privacy Principles (APPs) governing collection, use, disclosure, and storage of personal information. Serious data breaches must be notified under the Notifiable Data Breaches scheme.",
            
            "Health information is sensitive information under the Privacy Act with additional protections. The My Health Records Act 2012 governs the national digital health record system with specific privacy safeguards and access controls.",
        ]
        
        # Create embeddings
        print("🔄 Creating embeddings for demo documents...")
        embeddings = self.embedder.encode(self.documents)
        
        if FAISS_AVAILABLE:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
            self.index.add(embeddings)
            
            # Save FAISS index
            faiss.write_index(self.index, self.index_path)
        else:
            # Simple fallback - save embeddings
            self.embeddings = embeddings
            self.index = None
            with open(self.index_path.replace('.faiss', '_embeddings.pkl'), 'wb') as f:
                pickle.dump(embeddings, f)
        
        # Save documents
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        print(f"✅ Created demo index with {len(self.documents)} Australian legal documents")
    
    def _simple_search(self, query_embedding: np.ndarray, k: int) -> tuple:
        """Simple search without FAISS using cosine similarity"""
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_norm.T).flatten()
        
        # Get top k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_scores, top_k_indices.reshape(1, -1)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant legal documents"""
        try:
            # Encode query
            query_embedding = self.embedder.encode([query])
            
            # Search
            k = min(k, len(self.documents))
            
            if FAISS_AVAILABLE and self.index is not None:
                # Normalize query for FAISS
                faiss.normalize_L2(query_embedding)
                scores, indices = self.index.search(query_embedding, k)
                scores = scores[0]
                indices = indices[0]
            else:
                # Use simple search
                scores, indices_2d = self._simple_search(query_embedding[0], k)
                indices = indices_2d[0]
            
            # Load metadata if available
            metadata = []
            try:
                if os.path.exists("data/legal_metadata.pkl"):
                    with open("data/legal_metadata.pkl", 'rb') as f:
                        metadata = pickle.load(f)
            except:
                metadata = [{}] * len(self.documents)
            
            # Format results
            results = []
            for idx, score in zip(indices, scores):
                if idx < len(self.documents):
                    meta = metadata[idx] if idx < len(metadata) else {}
                    results.append({
                        'document': self.documents[idx],
                        'relevance_score': float(score),
                        'document_id': int(idx),
                        'jurisdiction': meta.get('jurisdiction', ''),
                        'doc_type': meta.get('type', ''),
                        'citation': meta.get('citation', ''),
                        'source': meta.get('source', ''),
                        'demo_mode': False  # Now using real data!
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            # Return a simple keyword-based fallback
            return self._fallback_search(query, k)
    
    def _fallback_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based fallback search"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Score documents by keyword overlap
        doc_scores = []
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())
            
            # Simple scoring: number of matching words
            score = len(query_words & doc_words) / max(len(query_words), 1)
            
            # Boost if exact phrase appears
            if query_lower in doc_lower:
                score += 0.5
            
            doc_scores.append((i, score, doc))
        
        # Sort by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for i, (idx, score, doc) in enumerate(doc_scores[:k]):
            if score > 0:  # Only return documents with some relevance
                results.append({
                    'document': doc,
                    'relevance_score': score,
                    'document_id': idx,
                    'demo_mode': True,
                    'fallback': True
                })
        
        return results