#!/bin/bash

echo "📦 Installing missing dependencies..."

# Core dependencies
pip install aiofiles aioredis websockets
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentence-transformers
pip install spacy networkx scikit-learn
pip install shap lime
pip install prometheus-client
pip install pandas numpy scipy
pip install chromadb faiss-cpu
pip install asyncpg sqlalchemy

# Download spaCy model
python -m spacy download en_core_web_lg

# Legal-specific models (create placeholder if not available)
echo "📚 Setting up legal models..."
mkdir -p models/legal-bert
cat > models/download_legal_bert.py << 'PYTHON'
from transformers import AutoModel, AutoTokenizer

print("Downloading legal-bert model...")
try:
    model = AutoModel.from_pretrained('nlpaueb/legal-bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    model.save_pretrained('./models/legal-bert')
    tokenizer.save_pretrained('./models/legal-bert')
    print("✓ Legal BERT downloaded successfully")
except Exception as e:
    print(f"Note: Using standard BERT as fallback: {e}")
    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model.save_pretrained('./models/legal-bert')
    tokenizer.save_pretrained('./models/legal-bert')
PYTHON

python models/download_legal_bert.py

echo "✅ Dependencies fixed!"
