
import sys
import time
import requests
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8002"
MAX_RETRIES = 10
RETRY_DELAY = 5

def log(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {message}")

def check_endpoint(endpoint, method="GET", payload=None, expected_status=200):
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=payload, timeout=10)
        else:
            return False, f"Unsupported method {method}"
            
        if response.status_code == expected_status:
            return True, response.json()
        else:
            return False, f"Status {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def wait_for_api():
    log(f"⏳ Waiting for API at {API_URL} to be ready...")
    for i in range(MAX_RETRIES):
        success, _ = check_endpoint("/health")
        if success:
            log("✅ API is online!", "SUCCESS")
            return True
        log(f"Process is starting... ({i+1}/{MAX_RETRIES})", "WAIT")
        time.sleep(RETRY_DELAY)
    
    log("❌ API failed to come online.", "ERROR")
    return False

def run_tests():
    if not wait_for_api():
        sys.exit(1)
        
    log("🚀 Starting API Verification Tests...")
    all_passed = True
    
    # Test 1: Health Check
    log("Test 1: Health Check", "TEST")
    success, data = check_endpoint("/health")
    if success:
        log(f"✅ Health OK: {data}", "PASS")
    else:
        log(f"❌ Health Check Failed: {data}", "FAIL")
        all_passed = False

    # Test 2: System Limits (if available)
    log("Test 2: Hardware Info", "TEST")
    success, data = check_endpoint("/research/system/hardware")
    if success:
        log(f"✅ Hardware Info: {json.dumps(data, indent=2)}", "PASS")
    else:
        log(f"⚠️ Hardware Info Endpoint not available (Optional)", "WARN")

    # Test 3: Ollama Connection (if available)
    log("Test 3: Ollama Connection", "TEST")
    success, data = check_endpoint("/test-ollama")
    if success:
        log(f"✅ Ollama Status: {json.dumps(data, indent=2)}", "PASS")
    else:
        log(f"❌ Ollama Test Failed: {data}", "FAIL")
        all_passed = False
        
    if all_passed:
        log("✨ All tests passed! API is fully functional.", "SUCCESS")
        sys.exit(0)
    else:
        log("⚠️ Some tests failed. Check logs.", "WARN")
        sys.exit(1)

if __name__ == "__main__":
    try:
        import requests
        run_tests()
    except ImportError:
        print("❌ 'requests' library not found. Please run: pip install requests")
        sys.exit(1)
