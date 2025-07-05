#!/bin/bash
# Run the optimized Legal AI API

echo "🚀 Starting Optimized Legal AI API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Add numpy import fix to existing files
echo "Fixing numpy imports in existing files..."
for file in ultimate_intelligent_legal_api.py ultimate_legal_ai_ultra.py ultimate_legal_ai_supreme.py; do
    if [ -f "$file" ] && ! grep -q "import numpy as np" "$file"; then
        sed -i '1s/^/import numpy as np\n/' "$file"
        echo "✅ Fixed numpy import in $file"
    fi
done

# Run the optimized standalone version
echo ""
echo "Starting API server..."
python optimized_main.py
