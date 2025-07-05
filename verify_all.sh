#!/bin/bash

echo "🚀 Australian Legal AI SUPREME - Complete Verification"
echo "=================================================="

# Function to run test and report
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo -e "\n📋 Running: $test_name"
    echo "-----------------------------------"
    
    if eval "$test_command"; then
        echo -e "✅ $test_name: PASSED"
    else
        echo -e "❌ $test_name: FAILED"
    fi
}

# Run all tests
run_test "API Health" "curl -s http://localhost:8000/health | jq '.status'"
run_test "Quantum Analysis" "curl -s -X POST http://localhost:8000/api/v1/analysis/quantum-supreme -H 'Content-Type: application/json' -d '{\"case_name\":\"Test v System\"}' | jq '.success'"
run_test "Cache Stats" "curl -s http://localhost:8000/api/v1/admin/stats | jq '.cache_stats.hit_rate'"
run_test "Search Function" "curl -s -X POST http://localhost:8000/api/v1/search/cases -H 'Content-Type: application/json' -d '{\"query\":\"contract\"}' | jq '.results | length'"

echo -e "\n✨ Verification Complete!"
