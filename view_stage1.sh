#!/bin/bash

echo "=== STAGE 1: Main System Files ==="

# Main unified system
echo -e "\n### 1. Main Unified System Fixed ###"
echo "File: unified_legal_ai_system_fixed.py"
echo "Size: $(wc -l unified_legal_ai_system_fixed.py 2>/dev/null | awk '{print $1}') lines"
echo "First 100 lines:"
head -100 unified_legal_ai_system_fixed.py

echo -e "\n### 2. Unified with Scraping ###"
echo "File: unified_with_scraping.py"
echo "Size: $(wc -l unified_with_scraping.py 2>/dev/null | awk '{print $1}') lines"
echo "First 100 lines:"
head -100 unified_with_scraping.py

