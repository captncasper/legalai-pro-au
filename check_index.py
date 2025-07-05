import numpy as np
# Save as check_index.py
import os
import json

if os.path.exists("data/index_stats.json"):
    with open("data/index_stats.json", "r") as f:
        stats = json.load(f)
    print("✅ Indexing COMPLETE!")
    print(f"📊 Total documents: {stats['total_documents']}")
    print(f"🏛️ Jurisdictions: {stats['jurisdictions']}")
    print(f"📄 Document types: {stats['document_types']}")
    print(f"📏 Avg doc length: {stats['average_doc_length']:.0f} chars")
else:
    print("❌ Index not found - still running or failed")
    
# Check file sizes
if os.path.exists("data/legal_index.faiss"):
    size_mb = os.path.getsize("data/legal_index.faiss") / 1024 / 1024
    print(f"\n💾 Index size: {size_mb:.2f} MB")