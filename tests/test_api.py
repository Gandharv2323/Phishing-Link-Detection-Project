"""
Test the Flask API endpoints
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

print("="*70)
print("TESTING HSEF WEB APP API")
print("="*70)

# Test 1: Health Check
print("\n1. Testing Health Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Model Info
print("\n2. Testing Model Info Endpoint...")
try:
    response = requests.get(f"{BASE_URL}/api/model_info")
    print(f"   Status: {response.status_code}")
    data = response.json()
    if response.status_code == 200:
        print(f"   Model Loaded: {data.get('loaded')}")
        print(f"   Classes: {data.get('classes')}")
        print(f"   Features: {data.get('n_features')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Single Prediction
print("\n3. Testing Single URL Prediction...")
try:
    payload = {"url": "https://test-example.com"}
    response = requests.post(
        f"{BASE_URL}/api/predict",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    print(f"   Status: {response.status_code}")
    data = response.json()
    if response.status_code == 200:
        print(f"   Prediction: {data.get('prediction')}")
        print(f"   Confidence: {data.get('confidence')*100:.2f}%")
    else:
        print(f"   Error: {data}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*70)
print("API TEST COMPLETE")
print("="*70)
print("\nIf all tests passed, the backend is working correctly.")
print("Issue is likely browser cache. Try:")
print("  1. Open browser in Incognito/Private mode")
print("  2. Hard refresh: CTRL+SHIFT+R or CTRL+F5")
print("  3. Clear browser cache for localhost")
