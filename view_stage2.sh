#!/bin/bash

echo "=== STAGE 2: API and Endpoints ==="

# API endpoints
echo -e "\n### 1. Case Upload Endpoints ###"
cat case_upload_endpoints.py

echo -e "\n### 2. Case Upload UI ###"
head -200 case_upload_ui.py

