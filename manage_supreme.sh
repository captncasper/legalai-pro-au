#!/bin/bash

case "$1" in
    start)
        echo "🚀 Starting Australian Legal AI SUPREME..."
        python3 legal_ai_supreme_au.py &
        echo $! > supreme.pid
        sleep 3
        
        if curl -s http://localhost:8000/health > /dev/null; then
            echo "✅ Legal AI SUPREME is running!"
            echo "📍 API Docs: http://localhost:8000/docs"
        else
            echo "❌ Failed to start"
        fi
        ;;
    
    stop)
        echo "⏹️  Stopping Legal AI SUPREME..."
        if [ -f supreme.pid ]; then
            kill $(cat supreme.pid) 2>/dev/null
            rm supreme.pid
        fi
        pkill -f "legal_ai_supreme_au.py" 2>/dev/null
        echo "✅ Stopped"
        ;;
    
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        if pgrep -f "legal_ai_supreme_au.py" > /dev/null; then
            echo "✅ Legal AI SUPREME is running"
            curl -s http://localhost:8000/health | python3 -m json.tool
        else
            echo "❌ Legal AI SUPREME is not running"
        fi
        ;;
    
    test)
        echo "🧪 Running tests..."
        python3 test_supreme.py
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|test}"
        exit 1
        ;;
esac
