#!/bin/bash
echo "🚀 Starting Optimized Legal AI API..."

# Check Python
python3 --version

# Install dependencies
pip install -r requirements.txt

# Fix numpy in existing files
for file in ultimate_intelligent_legal_api.py ultimate_legal_ai_ultra.py; do
    if [ -f "$file" ] && ! grep -q "import numpy" "$file"; then
        sed -i '1s/^/import numpy as np\n/' "$file"
        echo "✅ Fixed numpy in $file"
    fi
done

# Run the API
echo "Starting API server..."
python3 optimized_main.py
