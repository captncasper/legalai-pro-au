#!/bin/bash
echo "🔄 Migrating to Enhanced Legal AI..."

# Check current version
if pgrep -f "legal_ai_working.py" > /dev/null; then
    echo "✅ Found working version running"
    echo "⏹️  Stopping current version..."
    pkill -f "legal_ai_working.py"
    sleep 2
fi

# Backup current data
echo "💾 Backing up current configuration..."
cp legal_ai_working.py legal_ai_working.py.backup 2>/dev/null

# Start enhanced version
echo "🚀 Starting Enhanced Legal AI..."
python3 legal_ai_enhanced.py &

sleep 3

# Test if running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Enhanced API is running!"
    echo "📍 View docs at: http://localhost:8000/docs"
    echo "🧪 Run tests with: ./test_enhanced.py"
else
    echo "❌ Failed to start. Check logs."
fi
