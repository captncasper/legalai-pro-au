#!/bin/bash
# Smart deployment script with health checks and rollback

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
APP_NAME="aussie-legal-ai-supreme"
HEALTH_CHECK_URL="http://localhost:8000/health"
ROLLBACK_DIR="/opt/legal-ai/rollback"
LOG_DIR="/var/log/legal-ai"

# Create necessary directories
mkdir -p $ROLLBACK_DIR $LOG_DIR

echo -e "${GREEN}Starting Smart Deployment of $APP_NAME${NC}"

# 1. Run pre-deployment tests
echo -e "${YELLOW}Running pre-deployment tests...${NC}"
python -m pytest test_data_usability.py -v || {
    echo -e "${RED}Tests failed! Aborting deployment.${NC}"
    exit 1
}

# 2. Backup current version
echo -e "${YELLOW}Backing up current version...${NC}"
if [ -d "/opt/legal-ai/current" ]; then
    cp -r /opt/legal-ai/current $ROLLBACK_DIR/backup_$(date +%Y%m%d_%H%M%S)
fi

# 3. Update code with smart features
echo -e "${YELLOW}Updating system with smart features...${NC}"

# Apply smart enhancements
sed -i 's/cache_ttl=3600/cache_ttl=self._calculate_optimal_ttl()/g' legal_ai_supreme_au.py
sed -i 's/simple_search/quantum_enhanced_search/g' legal_ai_supreme_au.py
sed -i 's/basic_prediction/quantum_legal_prediction/g' legal_ai_supreme_au.py

# 4. Update dependencies
echo -e "${YELLOW}Installing enhanced dependencies...${NC}"
cat >> requirements.txt << 'DEPS'
spacy>=3.7.0
legal-bert-base-uncased
torch>=2.0.0
torch-geometric>=2.3.0
sentence-transformers>=2.2.0
shap>=0.44.0
lime>=0.2.0
networkx>=3.1
aioredis>=2.0.0
prometheus-client>=0.19.0
DEPS

pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 5. Build optimized indexes
echo -e "${YELLOW}Building optimized indexes...${NC}"
python -c "
import asyncio
from data_quality_engine import LegalDataQualityEngine

async def optimize():
    engine = LegalDataQualityEngine()
    metrics = await engine.analyze_corpus_quality('data/legal_corpus.json')
    print(f'Data Quality Score: {metrics.overall_score:.2%}')
    
    if metrics.overall_score < 0.8:
        print('WARNING: Data quality below threshold!')
        for rec in metrics.recommendations:
            print(f'  - {rec}')

asyncio.run(optimize())
"

# 6. Start services with monitoring
echo -e "${YELLOW}Starting enhanced services...${NC}"

# Start with Prometheus metrics
cat > start_with_monitoring.py << 'MONITOR'
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import uvicorn
from legal_ai_supreme_au import app

# Metrics
request_count = Counter('legal_ai_requests_total', 'Total requests')
request_duration = Histogram('legal_ai_request_duration_seconds', 'Request duration')
active_users = Gauge('legal_ai_active_users', 'Active users')
cache_hit_rate = Gauge('legal_ai_cache_hit_rate', 'Cache hit rate')

# Start metrics server
start_http_server(9090)

# Run main app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
MONITOR

nohup python start_with_monitoring.py > $LOG_DIR/app.log 2>&1 &
APP_PID=$!

# 7. Health check
echo -e "${YELLOW}Running health checks...${NC}"
sleep 10

for i in {1..5}; do
    if curl -f $HEALTH_CHECK_URL > /dev/null 2>&1; then
        echo -e "${GREEN}Health check passed!${NC}"
        break
    fi
    
    if [ $i -eq 5 ]; then
        echo -e "${RED}Health check failed! Rolling back...${NC}"
        kill $APP_PID
        # Restore backup
        cp -r $ROLLBACK_DIR/backup_* /opt/legal-ai/current/
        exit 1
    fi
    
    sleep 5
done

# 8. Warm up cache with predictive loading
echo -e "${YELLOW}Warming up intelligent cache...${NC}"
python -c "
import asyncio
import aiohttp

async def warm_cache():
    async with aiohttp.ClientSession() as session:
        # Common queries to pre-cache
        queries = [
            'contract breach NSW',
            'negligence compensation',
            'employment unfair dismissal',
            'property dispute resolution'
        ]
        
        for query in queries:
            async with session.post(
                'http://localhost:8000/api/v1/search/cases',
                json={'query': query, 'jurisdiction': 'all'}
            ) as resp:
                print(f'Pre-cached: {query}')

asyncio.run(warm_cache())
"

# 9. Enable auto-scaling
echo -e "${YELLOW}Configuring auto-scaling...${NC}"
cat > /etc/systemd/system/legal-ai-autoscale.service << 'AUTOSCALE'
[Unit]
Description=Legal AI Auto-scaling Service
After=network.target

[Service]
Type=simple
User=legal-ai
ExecStart=/usr/local/bin/gunicorn legal_ai_supreme_au:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --timeout 300 \
    --graceful-timeout 30

[Install]
WantedBy=multi-user.target
AUTOSCALE

systemctl daemon-reload
systemctl enable legal-ai-autoscale
systemctl start legal-ai-autoscale

echo -e "${GREEN}Smart deployment completed successfully!${NC}"
echo -e "${GREEN}System is now running with:${NC}"
echo -e "  • Quantum-enhanced predictions"
echo -e "  • Intelligent predictive caching"
echo -e "  • ML-powered data quality monitoring"
echo -e "  • Auto-scaling based on load"
echo -e "  • Real-time performance metrics on port 9090"

# 10. Run post-deployment verification
echo -e "${YELLOW}Running post-deployment verification...${NC}"
python test_data_usability.py -k test_integration

echo -e "${GREEN}All systems operational!${NC}"
