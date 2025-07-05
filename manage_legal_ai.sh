#!/bin/bash
# Legal AI Management Script

case "$1" in
    start)
        echo "🚀 Starting Legal AI..."
        if [ -f "legal_ai_enhanced.py" ]; then
            python3 legal_ai_enhanced.py
        else
            python3 legal_ai_working.py
        fi
        ;;
    
    stop)
        echo "⏹️  Stopping Legal AI..."
        pkill -f "legal_ai_"
        ;;
    
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    
    status)
        if pgrep -f "legal_ai_" > /dev/null; then
            echo "✅ Legal AI is running"
            curl -s http://localhost:8000/health | python3 -m json.tool
        else
            echo "❌ Legal AI is not running"
        fi
        ;;
    
    test)
        echo "🧪 Running tests..."
        if [ -f "test_enhanced.py" ]; then
            python3 test_enhanced.py
        else
            ./test_simple.sh
        fi
        ;;
    
    logs)
        echo "📋 Recent activity..."
        # Add log viewing logic here
        ;;
    
    *)
        echo "Usage: $0 {start|stop|restart|status|test|logs}"
        exit 1
        ;;
esac
