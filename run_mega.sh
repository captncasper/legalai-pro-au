#!/bin/bash
echo "🚀 Starting MEGA Legal AI API with ALL Features..."

# Install dependencies if needed
pip install -r requirements.txt 2>/dev/null

# Fix numpy in all files
for file in *.py; do
    if [ -f "$file" ] && ! grep -q "import numpy" "$file"; then
        sed -i '1s/^/import numpy as np\n/' "$file"
    fi
done

# Run the MEGA API
echo "Starting MEGA API server..."
python3 legal_ai_mega.py
