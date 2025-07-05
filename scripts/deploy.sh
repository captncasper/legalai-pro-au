#!/bin/bash
# Deploy to production

echo "🚀 Deploying Australian Legal AI..."

# Build Docker image
docker build -t aussie-legal-ai .

# Run with environment variables
docker run -d \
  --name legal-ai-api \
  -p 8000:8000 \
  -e API_KEY=$API_KEY \
  -v $(pwd)/data:/app/data \
  aussie-legal-ai

echo "✅ Deployed at http://localhost:8000"
