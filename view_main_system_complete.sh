#!/bin/bash

echo "=== Complete Main System Files ==="

# Show the rest of unified_legal_ai_system_fixed.py
echo -e "\n### Unified System Fixed - Lines 100-300 ###"
sed -n '100,300p' unified_legal_ai_system_fixed.py

echo -e "\n### Unified System Fixed - Lines 300-500 ###"
sed -n '300,500p' unified_legal_ai_system_fixed.py

echo -e "\n### Unified System Fixed - Last 100 lines ###"
tail -100 unified_legal_ai_system_fixed.py

echo -e "\n### Key Features Found ###"
echo "Endpoints:"
grep -E "@app\.(get|post|put|delete|websocket)" unified_legal_ai_system_fixed.py | head -20

echo -e "\nClasses:"
grep -E "^class " unified_legal_ai_system_fixed.py

