#!/bin/bash

BASE_URL="http://localhost:8000"
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "🔍 Verifying Australian Legal AI SUPREME Endpoints..."

# Test core endpoints
endpoints=(
    "/"
    "/health"
    "/docs"
    "/api/v1/admin/stats"
)

for endpoint in "${endpoints[@]}"; do
    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$endpoint")
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓${NC} $endpoint - OK"
    else
        echo -e "${RED}✗${NC} $endpoint - Failed (HTTP $response)"
    fi
done

# Test WebSocket
echo -e "\n�� Testing WebSocket connection..."
python3 -c "
import asyncio
import websockets

async def test_ws():
    try:
        async with websockets.connect('ws://localhost:8000/ws/legal-assistant') as ws:
            print('✓ WebSocket connected successfully')
    except Exception as e:
        print(f'✗ WebSocket failed: {e}')

asyncio.run(test_ws())
"
