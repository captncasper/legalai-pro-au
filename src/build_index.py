"""Build FAISS index from legal documents"""
import os
import sys
import torch
import faiss
import numpy as np
import pickle
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
from embeddings import LegalEmbedder


def main():
    print("🏗️ Building Australian Legal Search Index...")
    
    # Check for data
    data_path = os.environ.get('LEGAL_DATA_PATH', '/home/user/australian_legal_combined_finetune_data')
    if not os.path.exists(data_path):
        print(f"❌ Data not found at {data_path}")
        print("Please set LEGAL_DATA_PATH environment variable")
        sys.exit(1)
    
    # Load data
    print("📚 Loading legal documents...")
    dataset = load_from_disk(data_path)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    # Extract documents
    documents = []
    for i, item in enumerate(tqdm(dataset['train'], desc="Processing")):
        if i >= 10000:  # Start with 10k docs
            break
        text = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        documents.append(text[:2000])
    
    # Create embeddings
    print("🧮 Creating embeddings...")
    embedder = LegalEmbedder()
    embeddings = embedder.encode(documents)
    
    # Build index
    print("🔍 Building search index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    # Save
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/legal_index.faiss")
    with open("data/legal_documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    
    print(f"✅ Index built with {len(documents)} documents!")
    print("📁 Saved to data/legal_index.faiss")


if __name__ == "__main__":
    main()
