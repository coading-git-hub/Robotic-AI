#!/usr/bin/env python3
"""
Simple script to test if the unified API server endpoints are accessible.
"""
import time
import requests

def test_health_endpoint():
    print("Testing health endpoint with longer timeout...")

    try:
        response = requests.get("http://localhost:8002/api/health", timeout=10)
        print(f"[SUCCESS] Health endpoint accessible - Status: {response.status_code}")
        print(f"Response: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("[ERROR] Health endpoint - Connection refused (server not running)")
        return False
    except requests.exceptions.Timeout:
        print("[TIMEOUT] Health endpoint - Still initializing or not running")
        return False
    except Exception as e:
        print(f"[ERROR] Health endpoint - Error: {str(e)}")
        return False

def test_signup_endpoint():
    print("\nTesting signup endpoint...")

    try:
        # Try a GET request to see if the endpoint exists
        response = requests.get("http://localhost:8002/api/auth/signup", timeout=5)
        print(f"[SUCCESS] Signup endpoint accessible - Status: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("[ERROR] Signup endpoint - Connection refused (server not running)")
        return False
    except Exception as e:
        print(f"[ERROR] Signup endpoint - Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing unified API endpoints...")

    # Start the server in background first
    import subprocess
    import os
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    print("Starting unified API server in background...")
    server_process = subprocess.Popen(
        ["python", "-m", "src.main"],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait a bit for server to start
    time.sleep(3)

    # Test endpoints
    health_ok = test_health_endpoint()
    if health_ok:
        test_signup_endpoint()

    # Clean up
    try:
        server_process.terminate()
        server_process.wait(timeout=2)
    except:
        try:
            server_process.kill()
        except:
            pass