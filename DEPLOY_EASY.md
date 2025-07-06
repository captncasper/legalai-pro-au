# 🚀 Easy Railway Deployment (No CLI Required!)

## Option 1: Direct GitHub Deploy (EASIEST - No CLI Needed!)

### Step 1: Push to GitHub

```bash
# Initialize git and push your code
git init
git add .
git commit -m "Australian Legal AI - Professional Legal Document Generator"

# Create a new repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/legalai-pro-au.git
git push -u origin main
```

### Step 2: Deploy on Railway Website

1. **Go to Railway**: https://railway.app/new

2. **Click "Deploy from GitHub repo"**

3. **Connect your GitHub** and select your repo

4. **Add Environment Variable**:
   - Click on your deployed service
   - Go to "Variables" tab
   - Click "Add Variable"
   - Name: `HF_TOKEN`
   - Value: `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`
   - Click "Add"

5. **Wait for deployment** (usually 2-3 minutes)

6. **Get your URL**: Railway will show you something like:
   - `https://legalai-pro-au-production.up.railway.app`

That's it! No CLI needed! 🎉

---

## Option 2: Install Railway CLI Locally (If you prefer CLI)

```bash
# Install locally without sudo
npm install @railway/cli

# Use with npx
npx railway login
npx railway init
npx railway up
```

---

## Option 3: Use Sudo (Quick fix)

```bash
sudo npm install -g @railway/cli
```

---

## Direct Deploy Link

You can also use Railway's template deploy:

1. Make sure your code is on GitHub
2. Use this URL format:
   ```
   https://railway.app/new/github?repo=https://github.com/YOUR_USERNAME/legalai-pro-au
   ```

---

## Quick GitHub Push Commands

If you haven't created a GitHub repo yet:

```bash
# 1. Create a new repository on GitHub.com (don't initialize with README)

# 2. In your terminal:
git init
git add .
git commit -m "Australian Legal AI - Professional Legal Document Generator"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/legalai-pro-au.git
git push -u origin main
```

---

## After Deployment

Your app will be available at:
- Main App: `https://your-app.railway.app/`
- Health Check: `https://your-app.railway.app/health`

Remember to add your HuggingFace token in Railway's environment variables!