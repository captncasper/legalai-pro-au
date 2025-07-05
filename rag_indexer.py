import numpy as np
#!/usr/bin/env python3
"""
Legal RAG System - Build semantic index with citations
"""

import pickle
from sentence_transformers import SentenceTransformer
import chromadb
import re
from tqdm import tqdm

print("🚀 Building Legal RAG System...")

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
print("Initializing vector database...")
client = chromadb.PersistentClient(path="./rag_index")
collection = client.get_or_create_collection(
    name="aussie_legal",
    metadata={"hnsw:space": "cosine"}
)

# Load your documents
print("Loading legal documents...")
with open('data/simple_index.pkl', 'rb') as f:
    data = pickle.load(f)
    documents = data['documents']

print(f"Processing {len(documents)} documents...")

# Process in batches
batch_size = 100
for i in tqdm(range(0, len(documents), batch_size)):
    batch = documents[i:i+batch_size]
    
    texts = []
    metadatas = []
    ids = []
    
    for j, doc in enumerate(batch):
        # Split into chunks
        text = doc['text']
        chunks = [text[k:k+500] for k in range(0, len(text), 400)]
        
        for k, chunk in enumerate(chunks[:5]):  # Max 5 chunks per doc
            texts.append(chunk)
            ids.append(f"doc_{i+j}_chunk_{k}")
            metadatas.append({
                'citation': doc.get('metadata', {}).get('citation', 'Unknown'),
                'type': doc.get('metadata', {}).get('type', 'case'),
                'date': doc.get('metadata', {}).get('date', ''),
                'chunk': k,
                'doc_id': i+j
            })
    
    # Embed and add to collection
    embeddings = model.encode(texts)
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

print(f"✅ RAG index created with {collection.count()} chunks!")
