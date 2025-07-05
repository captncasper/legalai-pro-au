#!/bin/bash
echo "🚀 Starting Legal AI API..."

# Install basic requirements
pip install fastapi uvicorn numpy pydantic

# Fix numpy imports
for file in *.py; do
    if [ -f "$file" ] && ! grep -q "import numpy" "$file"; then
        echo "import numpy as np" | cat - "$file" > temp && mv temp "$file"
    fi
done

# Try different versions in order
if [ -f "legal_ai_working.py" ]; then
    echo "Starting working version..."
    python3 legal_ai_working.py
elif [ -f "optimized_main.py" ]; then
    echo "Starting optimized version..."
    python3 optimized_main.py
else
    echo "Starting any available version..."
    python3 ultimate_intelligent_legal_api.py || python3 legal_qa_light.py
fi
