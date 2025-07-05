import numpy as np
#!/usr/bin/env python3
"""
Fixed Legal RAG System - Handles missing metadata
"""

import pickle
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

print("🚀 Building Legal RAG System...")

# Load documents
print("Loading legal documents...")
with open('data/simple_index.pkl', 'rb') as f:
    data = pickle.load(f)
    documents = data['documents']

print(f"Found {len(documents)} documents")

# Setup ChromaDB
client = chromadb.PersistentClient(path="./rag_index")

# Reset collection
try:
    client.delete_collection("aussie_legal")
except:
    pass

collection = client.create_collection(
    name="aussie_legal",
    metadata={"hnsw:space": "cosine"}
)

# Load model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Process documents
print("Indexing documents...")
batch_size = 50
total_indexed = 0

for i in tqdm(range(0, min(2000, len(documents)), batch_size)):  # First 2000 docs
    batch = documents[i:i+batch_size]
    
    texts = []
    metadatas = []
    ids = []
    
    for j, doc in enumerate(batch):
        text = doc['text']
        
        # Handle long texts - chunk them
        chunks = [text[k:k+500] for k in range(0, min(len(text), 2000), 400)]
        
        for k, chunk in enumerate(chunks[:3]):  # Max 3 chunks per doc
            if len(chunk.strip()) < 10:  # Skip empty chunks
                continue
                
            texts.append(chunk)
            ids.append(f"doc_{i+j}_chunk_{k}")
            
            # Fix metadata - ensure no None values
            metadata = doc.get('metadata', {})
            clean_metadata = {
                'citation': str(metadata.get('citation', f'Document {i+j}')),
                'type': str(metadata.get('type', 'legal_document')),
                'date': str(metadata.get('date', 'unknown')),
                'jurisdiction': str(metadata.get('jurisdiction', 'australia')),
                'doc_id': str(i+j),
                'chunk_id': str(k)
            }
            
            # Ensure no None values
            for key, value in clean_metadata.items():
                if value is None or value == 'None':
                    clean_metadata[key] = 'unknown'
            
            metadatas.append(clean_metadata)
    
    if texts:  # Only add if we have texts
        # Embed and add
        embeddings = model.encode(texts, show_progress_bar=False)
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        total_indexed += len(texts)

print(f"✅ Indexed {total_indexed} chunks from {min(2000, len(documents))} documents!")

# Test the index
print("\n🧪 Testing the index...")
test_queries = ["unfair dismissal", "contract breach", "negligence"]

for query in test_queries:
    test_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[test_embedding.tolist()],
        n_results=2
    )
    
    print(f"\nQuery: '{query}'")
    if results['documents'][0]:
        print(f"Found {len(results['documents'][0])} results")
        print(f"First result: {results['documents'][0][0][:100]}...")
        print(f"Citation: {results['metadatas'][0][0]['citation']}")
    else:
        print("No results found")

print("\n✅ RAG index ready to use!")
