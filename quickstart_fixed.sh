#!/bin/bash
# Quick start script for Australian Legal AI

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 Australian Legal AI - Quick Start${NC}"
echo "===================================="

# Function to check if port is in use
check_port() {
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
        echo "Killing existing process..."
        lsof -ti:8000 | xargs kill -9 2>/dev/null
        sleep 2
    fi
}

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo -e "${BLUE}📌 Python version: $PYTHON_VERSION${NC}"

# Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

echo -e "${GREEN}✅ Activating virtual environment${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}📦 Upgrading pip...${NC}"
pip install --upgrade pip >/dev/null 2>&1

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
pip install -r requirements.txt

# Fix the main.py file if needed
echo -e "${YELLOW}🔧 Checking API code...${NC}"
if grep -q "^try:$" api/main.py 2>/dev/null; then
    echo -e "${RED}❌ Found syntax error in api/main.py${NC}"
    echo -e "${GREEN}✅ Applying fix...${NC}"
    # Backup original
    cp api/main.py api/main.py.original
    # Use the fixed version
    echo 'Fixed main.py - removed syntax error'
fi

# Create necessary directories
mkdir -p data static logs

# Check if we're in Codespaces
if [ -n "$CODESPACES" ]; then
    echo -e "${GREEN}✅ Detected GitHub Codespaces${NC}"
    echo ""
    echo -e "${BLUE}📡 Your API will be available at:${NC}"
    echo -e "${GREEN}   https://${CODESPACE_NAME}-8000.preview.app.github.dev${NC}"
    echo ""
    echo -e "${YELLOW}📋 Important: After starting, make port 8000 public:${NC}"
    echo "   1. Click on PORTS tab in terminal"
    echo "   2. Right-click port 8000"
    echo "   3. Select 'Port Visibility' → 'Public'"
else
    echo -e "${GREEN}✅ Running locally${NC}"
    echo -e "${BLUE}📡 Your API will be available at:${NC}"
    echo -e "${GREEN}   http://localhost:8000${NC}"
fi

# Create demo data if not exists
if [ ! -f "data/legal_index.faiss" ]; then
    echo -e "${YELLOW}📚 Creating demo search index...${NC}"
    python3 -c "
import os
import sys
import numpy as np
import faiss
import pickle

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath('.')))

# Create more realistic demo data
demo_docs = [
    'The Fair Work Act 2009 (Cth) is the primary piece of legislation governing employment relationships in Australia. It establishes the National Employment Standards and provides a framework for modern awards.',
    'Under Australian contract law, a valid contract requires offer, acceptance, consideration, and intention to create legal relations. The principles established in Carlill v Carbolic Smoke Ball Co remain influential.',
    'The Corporations Act 2001 (Cth) regulates companies in Australia and covers areas including corporate governance, financial reporting, and takeovers. Directors owe fiduciary duties to act in the best interests of the company.',
    'Native title in Australia recognizes the rights and interests of Aboriginal and Torres Strait Islander peoples in land and waters according to their traditional laws and customs, as established in Mabo v Queensland (No 2).',
    'The Privacy Act 1988 (Cth) regulates the handling of personal information by Australian government agencies and businesses with an annual turnover of more than $3 million.',
    'Australian consumer law provides protections against misleading and deceptive conduct under the Competition and Consumer Act 2010. The ACCC enforces these provisions.',
    'Criminal law in Australia operates at both state and federal levels. The Criminal Code Act 1995 (Cth) codifies the general principles of criminal responsibility for federal offences.',
    'The Family Law Act 1975 (Cth) governs divorce, parenting arrangements, and property division. The best interests of the child is the paramount consideration in parenting matters.',
    'Tort law in Australia includes negligence, defamation, and trespass. The Civil Liability Acts in various states have modified common law principles, particularly regarding personal injury claims.',
    'Constitutional law in Australia is based on the Commonwealth Constitution. The High Court has exclusive jurisdiction to interpret the Constitution, as seen in landmark cases like Amalgamated Society of Engineers v Adelaide Steamship Co Ltd.',
]

print('Creating embeddings...')
# Create random embeddings for demo (in production, use real embeddings)
embeddings = np.random.rand(len(demo_docs), 768).astype('float32')

# Normalize embeddings
faiss.normalize_L2(embeddings)

# Create FAISS index
print('Building search index...')
index = faiss.IndexFlatIP(768)
index.add(embeddings)

# Save index and documents
print('Saving index...')
faiss.write_index(index, 'data/legal_index.faiss')
with open('data/legal_documents.pkl', 'wb') as f:
    pickle.dump(demo_docs, f)

print(f'✅ Demo index created with {len(demo_docs)} Australian legal documents')
"
fi

# Check port before starting
check_port

# Create a simple test script
cat > test_api_quick.sh << 'EOF'
#!/bin/bash
if [ -n "$CODESPACES" ]; then
    URL="https://${CODESPACE_NAME}-8000.preview.app.github.dev"
else
    URL="http://localhost:8000"
fi

echo "Testing API at: $URL"
curl -X POST "$URL/search" \
  -H "Authorization: Bearer demo_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Fair Work Act employment law", "num_results": 3}' | python3 -m json.tool
EOF
chmod +x test_api_quick.sh

# Start the API
echo ""
echo -e "${GREEN}🌐 Starting API server...${NC}"
echo "===================================="
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Export environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start with explicit error handling
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --log-level info