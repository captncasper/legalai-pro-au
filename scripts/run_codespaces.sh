#!/bin/bash
# Run in GitHub Codespaces with proper port configuration

echo "�� Starting Australian Legal AI in Codespaces..."

# Detect if we're in Codespaces
if [ -n "$CODESPACES" ]; then
    echo "✅ Detected GitHub Codespaces environment"
    echo ""
    echo "⚠️  IMPORTANT: After starting, you need to:"
    echo "1. Go to the PORTS tab in the terminal panel"
    echo "2. Find port 8000"
    echo "3. Right-click and select 'Port Visibility' → 'Public'"
    echo ""
    echo "Your API will be available at:"
    echo "https://$CODESPACE_NAME-8000.preview.app.github.dev"
    echo ""
    
    # Start with host 0.0.0.0 for external access
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Running in standard environment..."
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
fi
