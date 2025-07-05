#!/bin/bash

echo "📊 Monitoring System Metrics..."

# Check if Prometheus metrics are available
if curl -s http://localhost:9090/metrics > /dev/null 2>&1; then
    echo "✓ Prometheus metrics available at http://localhost:9090/metrics"
    
    # Sample some key metrics
    curl -s http://localhost:9090/metrics | grep -E "(legal_ai_requests_total|legal_ai_cache_hit_rate|legal_ai_request_duration)" | head -10
else
    echo "ℹ️  Prometheus metrics not configured yet"
fi

# Check system resources
echo -e "\n💻 System Resources:"
ps aux | grep -E "(legal_ai|uvicorn)" | grep -v grep
echo -e "\n📈 Memory Usage:"
free -h | grep -E "(Mem|Swap)"
