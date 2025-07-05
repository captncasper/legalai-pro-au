#!/bin/bash

echo "🚀 Enhancing Australian Legal AI Data"
echo "===================================="

# 1. Analyze existing code
echo -e "\n1️⃣ Analyzing existing code for reusable features..."
python analyze_all_code.py

# 2. Extract settlement amounts
echo -e "\n2️⃣ Extracting settlement amounts from corpus..."
python extract_settlement_amounts.py

# 3. Analyze judge patterns
echo -e "\n3️⃣ Analyzing judge behavior patterns..."
python analyze_judge_patterns.py

# 4. Get more cases (optional)
echo -e "\n4️⃣ Checking AustLII for new cases..."
python scrape_austlii.py

echo -e "\n✅ Data enhancement complete!"
echo "Check the generated files:"
echo "  - code_analysis_report.json"
echo "  - Settlement amounts extracted"
echo "  - Judge patterns analyzed"
echo "  - New cases available in austlii_cases_new.json"
