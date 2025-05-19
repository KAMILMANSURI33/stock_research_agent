import requests
import json
import time

def test_api():
    url = "http://127.0.0.1:8000/analyze"
    headers = {"Content-Type": "application/json"}
    data = {
        "symbol": "AAPL",
        "days": 3
    }
    
    print(f"Sending request to {url}")
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            print(f"\nAttempt {attempt + 1}:")
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print("\nProcessed response:")
                print(json.dumps(result, indent=2))
                return
            else:
                print(f"Response body: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"\nConnection failed on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("Failed to connect after all retries")
        except Exception as e:
            print(f"\nError on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print("Failed after all retries")

if __name__ == "__main__":
    test_api() 