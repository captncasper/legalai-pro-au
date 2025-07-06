#!/bin/bash

echo "🚀 Deploying Australian Legal AI to Railway"
echo "==========================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Please install it first:"
    echo "   npm install -g @railway/cli"
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not set. You'll need to add it in Railway dashboard."
    echo "   Get your token from: https://huggingface.co/settings/tokens"
fi

# Initialize git if needed
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Australian Legal AI"
fi

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Link or create project
echo "🔗 Linking Railway project..."
railway link || railway init

# Deploy
echo "🚀 Deploying to Railway..."
railway up

# Add environment variables
if [ ! -z "$HF_TOKEN" ]; then
    echo "🔑 Setting HuggingFace token..."
    railway variables set HF_TOKEN="$HF_TOKEN"
fi

# Get deployment URL
echo "✅ Deployment complete!"
echo "🌐 Your app will be available at:"
railway open

echo ""
echo "📝 Next steps:"
echo "1. Add HF_TOKEN in Railway dashboard if not set"
echo "2. Wait 1-2 minutes for deployment to complete"
echo "3. Visit your app URL to test the Legal AI"