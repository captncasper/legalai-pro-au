#!/bin/bash

echo "=== Viewing Core Application Files ==="

# Main unified system (most recent)
echo -e "\n### 1. Main Unified System ###"
cat unified_legal_ai_system_fixed.py

# API endpoints and UI
echo -e "\n### 2. Case Upload Endpoints ###"
cat case_upload_endpoints.py

echo -e "\n### 3. Case Upload UI ###"
cat case_upload_ui.py

# Scraping components
echo -e "\n### 4. Intelligent Legal Scraper ###"
cat intelligent_legal_scraper.py

echo -e "\n### 5. Alternative Scrapers ###"
cat alternative_scrapers.py

# Latest integration files
echo -e "\n### 6. Unified System with Scraping ###"
cat unified_with_scraping.py

# Machine Learning components
echo -e "\n### 7. Outcome Predictor Training ###"
cat train_outcome_predictor.py

echo -e "\n### 8. Quantum Legal Predictor ###"
cat quantum_legal_predictor.py

# Data quality and intelligence
echo -e "\n### 9. Data Quality Engine ###"
cat data_quality_engine.py

echo -e "\n### 10. Intelligent Cache Manager ###"
cat intelligent_cache_manager.py

# Test files to understand functionality
echo -e "\n### 11. Test New Features ###"
cat test_new_features.py

# Configuration and requirements
echo -e "\n### 12. Requirements ###"
cat requirements.txt

# Check for environment configuration
echo -e "\n### 13. Environment Example ###"
if [ -f .env.example ]; then
    cat .env.example
fi

# Recent integration script
echo -e "\n### 14. Integration Script ###"
cat integrate_alternative_scrapers.py

# Startup scripts
echo -e "\n### 15. Unified System Startup ###"
cat start_unified_fixed.sh

# Check app structure
echo -e "\n### 16. App Directory Structure ###"
if [ -d app ]; then
    ls -la app/
fi

# Check for main entry points
echo -e "\n### 17. Main Entry Points ###"
ls -la *.py | grep -E "(main|app|server)" | head -10

