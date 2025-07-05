import numpy as np
import requests

# Test local first
local_url = "http://localhost:8000/health"
response = requests.get(local_url)
print(f"Local health check: {response.status_code}")
print(response.text)

# Test search locally
search_url = "http://localhost:8000/search"
headers = {"Authorization": "Bearer demo_key"}
data = {"query": "contract law", "num_results": 3}

response = requests.post(search_url, json=data, headers=headers)
print(f"\nLocal search status: {response.status_code}")
print(response.text)