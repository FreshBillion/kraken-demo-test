import requests
import time

def run_test():
    try:
        r = requests.get(
            "https://demo-futures.kraken.com/derivatives/api/v3/tickers",
            headers={"Accept": "application/json"},
            timeout=15
        )
        print(f"Status: {r.status_code}")
        print(f"Content-Type: {r.headers.get('Content-Type')}")
        print(r.text[:500])
    except Exception as e:
        print(f"Request failed: {e}")

run_test()
# Keep the process alive so Railway doesn't mark it as "crashed" —
# you'll just check the logs, then delete this service when done.
while True:
    time.sleep(3600)
