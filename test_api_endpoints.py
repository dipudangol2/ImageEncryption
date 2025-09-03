#!/usr/bin/env python3
"""
Test script to verify API endpoints work with password-based encryption
"""

import requests
import json
from pathlib import Path
import time

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test that the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health Check: {data['message']}")
            return True
        else:
            print(f"❌ API Health Check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health Check failed: {e}")
        return False

def test_password_validation():
    """Test password validation with different lengths"""
    print("\nTesting Password Validation")
    print("-" * 40)
    
    # Test with sample image (we'll create a simple test image)
    test_image_path = Path("sample_image.jpg")
    
    if not test_image_path.exists():
        print("❌ No test image found. Please ensure sample_image.jpg exists for testing.")
        return
    
    # Test passwords of different lengths
    test_passwords = [
        "",  # Empty password (should fail)
        "short",  # Short password (should work)
        "medium_length_password",  # Medium password (should work)
        "this_is_a_very_long_password_that_exceeds_16_characters_by_far",  # Long password (should work)
        "🔒 Unicode! 🔑",  # Unicode password (should work)
    ]
    
    for i, password in enumerate(test_passwords):
        print(f"\nTest {i+1}: Password length = {len(password)}")
        print(f"Password: '{password[:30]}{'...' if len(password) > 30 else ''}'")
        
        try:
            with open(test_image_path, 'rb') as f:
                files = {'image': f}
                data = {'key': password, 'quality': '75'}
                
                response = requests.post(
                    f"{API_BASE_URL}/api/encrypt", 
                    files=files, 
                    data=data, 
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print(f"✅ Encryption successful with password length {len(password)}")
                    else:
                        print(f"❌ Encryption failed: {result.get('message', 'Unknown error')}")
                elif response.status_code == 400:
                    error_data = response.json()
                    print(f"⚠️  Expected validation error: {error_data['detail']}")
                else:
                    print(f"❌ Unexpected error: {response.status_code}")
                    
        except Exception as e:
            print(f"❌ Request failed: {e}")

def main():
    print("Testing Password-Based Image Encryption API")
    print("=" * 50)
    
    # Test if API is running
    if not test_health_endpoint():
        print("\n❌ API is not running. Please start the backend server first:")
        print("   cd api && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    # Test password validation
    test_password_validation()
    
    print("\n" + "=" * 50)
    print("API Password System Test Complete!")

if __name__ == "__main__":
    main()