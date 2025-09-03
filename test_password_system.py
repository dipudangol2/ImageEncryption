#!/usr/bin/env python3
"""
Test script to verify SHA-256 password-based key derivation system
"""

import sys
import hashlib
from pathlib import Path

# Add the api directory to path
sys.path.append(str(Path(__file__).parent / "api"))

from api.aes_cipher import AESCipher
from api.main import derive_aes_key

def test_key_derivation():
    """Test SHA-256 key derivation with different password lengths"""
    
    print("Testing SHA-256 Key Derivation System")
    print("=" * 50)
    
    # Test passwords of different lengths
    test_passwords = [
        "short",
        "medium_password",
        "this_is_a_very_long_password_with_many_characters",
        "MySecurePassword123!",
        "🔒🔑 Unicode password with emojis 🛡️",
        "a" * 100,  # Very long password
    ]
    
    for i, password in enumerate(test_passwords, 1):
        print(f"\nTest {i}: Password length = {len(password)} characters")
        print(f"Password: '{password[:50]}{'...' if len(password) > 50 else ''}'")
        
        # Derive key using our function
        derived_key = derive_aes_key(password)
        print(f"Derived key (hex): {derived_key.hex()}")
        print(f"Key length: {len(derived_key)} bytes")
        
        # Test if AESCipher can be initialized with derived key
        try:
            cipher = AESCipher(derived_key)
            print("✅ AESCipher initialized successfully")
            
            # Test encryption/decryption round trip
            test_data = b"Hello, this is a test message for encryption!"
            encrypted = cipher.encrypt(test_data)
            decrypted = cipher.decrypt(encrypted)
            
            if decrypted == test_data:
                print("✅ Encryption/decryption round trip successful")
            else:
                print("❌ Encryption/decryption round trip failed")
                
        except Exception as e:
            print(f"❌ Error with AESCipher: {e}")
    
    print("\n" + "=" * 50)
    print("Key Derivation Test Complete!")

def test_consistency():
    """Test that same password always produces same key"""
    print("\nTesting Key Derivation Consistency")
    print("-" * 40)
    
    password = "MyTestPassword123"
    
    # Generate key multiple times
    keys = [derive_aes_key(password) for _ in range(5)]
    
    # Check all keys are identical
    if all(key == keys[0] for key in keys):
        print(f"✅ Same password produces consistent key: {keys[0].hex()}")
    else:
        print("❌ Key derivation is inconsistent!")
        for i, key in enumerate(keys):
            print(f"  Attempt {i+1}: {key.hex()}")

if __name__ == "__main__":
    test_key_derivation()
    test_consistency()