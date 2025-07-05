#!/bin/bash

echo "=== STAGE 4: ML and Intelligence Components ==="

# ML components
echo -e "\n### 1. Quantum Legal Predictor ###"
head -150 quantum_legal_predictor.py

echo -e "\n### 2. Train Outcome Predictor ###"
cat train_outcome_predictor.py

echo -e "\n### 3. Semantic Search ###"
if [ -f add_semantic_search.py ]; then
    cat add_semantic_search.py
fi

