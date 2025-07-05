#!/bin/bash

echo "=== Integration Points and Architecture ==="

# Check for scraping integration
echo -e "\n### Scraping Integration ###"
if [ -f add_scraping_to_api.py ]; then
    cat add_scraping_to_api.py
fi

echo -e "\n### Database/Models Structure ###"
if [ -d models ]; then
    ls -la models/
    # Check for model files
    for file in models/*.py; do
        if [ -f "$file" ]; then
            echo -e "\n--- $file ---"
            head -50 "$file"
        fi
    done
fi

echo -e "\n### Check for Configuration ###"
for config in *.json *.yaml *.yml .env; do
    if [ -f "$config" ]; then
        echo -e "\n--- $config ---"
        head -20 "$config"
    fi
done

echo -e "\n### API Structure ###"
if [ -d api ]; then
    find api -name "*.py" -type f | head -10
fi

echo -e "\n### Recent Log Files ###"
if [ -d logs ]; then
    ls -la logs/ | head -10
fi

