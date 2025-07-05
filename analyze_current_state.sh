#!/bin/bash

echo "=== Current System Analysis ==="

echo -e "\n### Python Files Summary ###"
echo "Total Python files: $(find . -name "*.py" -type f | wc -l)"
echo "Main entry points:"
ls -la *.py | grep -E "(main|app|server|api)" | awk '{print $9, "-", $5, "bytes"}'

echo -e "\n### Recent Updates ###"
echo "Files modified today:"
find . -name "*.py" -mtime -1 -type f | head -10

echo -e "\n### API Endpoints Summary ###"
echo "From unified_legal_ai_system_fixed.py:"
grep -E "@app\.(get|post|put|delete|websocket)" unified_legal_ai_system_fixed.py | wc -l
echo "endpoints found"

echo -e "\n### External Dependencies ###"
if [ -f requirements.txt ]; then
    echo "Key packages:"
    grep -E "(fastapi|torch|transformers|sentence-transformers|pandas|numpy)" requirements.txt
fi

echo -e "\n### Scraping Components ###"
ls -la *scrap*.py 2>/dev/null | wc -l
echo "scraping-related files"

echo -e "\n### ML Components ###"
ls -la *predict*.py *train*.py *model*.py 2>/dev/null | wc -l
echo "ML-related files"

echo -e "\n### What would you like to update? ###"
echo "1. Add new features (what kind?)"
echo "2. Improve existing functionality"
echo "3. Fix bugs or issues"
echo "4. Optimize performance"
echo "5. Add new data sources"
echo "6. Enhance UI/UX"
echo "7. Other improvements"

