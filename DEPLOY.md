# 🚀 Deploying Australian Legal AI to Railway

## Prerequisites
- Railway account (https://railway.app)
- GitHub repository with this code
- HuggingFace account and API token

## Step 1: Environment Setup

1. Get your HuggingFace token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with read permissions
   - Copy the token (starts with `hf_`)

## Step 2: Deploy to Railway

### Option A: Deploy from GitHub (Recommended)

1. Push this code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Australian Legal AI - Professional Legal Document Generator"
   git remote add origin https://github.com/YOUR_USERNAME/aussie-legal-ai.git
   git push -u origin main
   ```

2. Connect to Railway:
   - Go to https://railway.app/new
   - Click "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect the configuration

3. Add Environment Variables:
   - In Railway dashboard, go to your project
   - Click on the service
   - Go to "Variables" tab
   - Add: `HF_TOKEN` = your HuggingFace token
   - Railway will automatically set `PORT`

### Option B: Deploy via Railway CLI

1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login and deploy:
   ```bash
   railway login
   railway init
   railway up
   railway add --name HF_TOKEN --value "your_hf_token_here"
   ```

## Step 3: Access Your App

Once deployed, Railway will provide you with a URL like:
- `https://aussie-legal-ai-production.up.railway.app`

## Features Available

### Main App (Professional Legal Brief Generator)
- URL: `https://your-app.railway.app/`
- Generate court-ready legal documents
- Professional Statement of Claim creation
- Multiple jurisdictions and document types

### Simple Case Predictor
- URL: `https://your-app.railway.app/simple`
- Quick case outcome predictions
- Success probability analysis

### Full Feature Set (All Tools)
- URL: `https://your-app.railway.app/full`
- Access to all experimental features

## API Endpoints

- `POST /api/v1/generate-legal-brief` - Generate legal documents
- `POST /api/v1/predict-outcome` - Predict case outcomes
- `POST /api/v1/analyze-risk` - Risk analysis
- `POST /api/v1/generate-strategy` - Legal strategy generation
- `POST /api/v1/predict-settlement` - Settlement predictions

## Performance Notes

- First load may take 30-60 seconds (loading AI models)
- Subsequent requests will be much faster
- Railway's free tier includes 500 hours/month
- Upgrade to Railway Pro for production use

## Troubleshooting

1. **Out of Memory**: The AI models require ~2GB RAM
   - Solution: Upgrade to Railway Pro for more resources

2. **Slow Initial Load**: Normal behavior while loading models
   - The app caches models after first load

3. **HuggingFace Token Issues**: 
   - Ensure token starts with `hf_`
   - Check token has read permissions

## Production Checklist

- [ ] Set strong HuggingFace token
- [ ] Configure custom domain (optional)
- [ ] Set up monitoring/alerts
- [ ] Enable auto-scaling for high traffic
- [ ] Review rate limiting settings
- [ ] Set up backup/recovery plan

## Support

For issues or questions:
- Railway Discord: https://discord.gg/railway
- Railway Docs: https://docs.railway.app

## Cost Estimates

- Railway Hobby: $5/month (500 hours)
- Railway Pro: $20/month (unlimited)
- Recommended for production: Railway Pro

Your Australian Legal AI is now ready for deployment! 🎉