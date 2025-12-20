#!/usr/bin/env python3
"""
Check what routes are available on the running server.
"""
import requests

def check_routes():
    print("Checking available routes via API docs...")

    # Try to access the OpenAPI docs
    try:
        response = requests.get("http://localhost:8002/openapi.json", timeout=10)
        print(f"OpenAPI JSON endpoint - Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"API Title: {data.get('info', {}).get('title', 'Unknown')}")
            print(f"Available paths: {list(data.get('paths', {}).keys())}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error accessing OpenAPI JSON: {e}")

if __name__ == "__main__":
    check_routes()