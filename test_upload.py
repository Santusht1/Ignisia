import requests
import sys

try:
    with open('data/sample_sales.csv', 'rb') as f:
        files = {'file': f}
        r = requests.post('http://127.0.0.1:8000/api/upload', files=files)
    
    if r.status_code == 200:
        data = r.json()
        print("Success!")
        print("Best Model:", data.get('best_model'))
        print("Metrics:", list(data.get('metrics', {}).keys()))
        print("Evaluated Models (from metrics keys):", list(data.get('metrics', {}).get('Linear Regression', {}).keys()) if 'metrics' in data else 'N/A')
        # Wait, the API only returns the metrics for the best model.
        print(data.keys())
    else:
        print("Error:", r.status_code, r.text)
except Exception as e:
    print("Exception:", e)
