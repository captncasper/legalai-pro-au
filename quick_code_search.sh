#!/bin/bash

echo "🔍 Searching for key implementations in your code..."
echo "=" * 60

echo -e "\n💰 Settlement Amount Extraction:"
grep -l -i "settlement.*amount\|\$[0-9]" *.py | head -5

echo -e "\n👨‍⚖️ Judge Analysis:"
grep -l -i "judge.*pattern\|judge.*analysis" *.py | head -5

echo -e "\n🌐 AustLII Integration:"
grep -l -i "austlii\|scrape.*case" *.py | head -5

echo -e "\n🧠 ML Models:"
grep -l -i "predict.*outcome\|RandomForest\|neural" *.py | head -5

echo -e "\n🔍 Semantic Search:"
grep -l -i "embedding\|semantic.*search\|sentence.*transformer" *.py | head -5

echo -e "\n📊 Data Extraction:"
grep -l -i "extract.*from.*text\|parse.*document" *.py | head -5
