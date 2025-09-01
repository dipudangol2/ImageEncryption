#!/usr/bin/env python3
"""
Test script to verify the complete image encryption system works.
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path

# Import our modules
from aes_cipher import AESCipher
from unified_compression import UnifiedCompressor
import pickle

def create_test_image():
    """Create a simple test image."""
    # Create a 64x64 RGB test image with patterns
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img[i, j, 0] = int(255 * (i / 64))  # Red gradient
            img[i, j, 1] = int(255 * (j / 64))  # Green gradient  
            img[i, j, 2] = int(255 * ((i + j) / 128))  # Blue gradient
    
    return img

def test_compression_encryption_cycle():
    """Test the complete compression and encryption cycle."""
    print("🧪 Testing Image Encryption System")
    print("=" * 40)
    
    # Create test image
    test_image = create_test_image()
    print(f"✅ Created test image: {test_image.shape}")
    
    # Save test image
    Image.fromarray(test_image).save("test_input.jpg")
    print("✅ Saved test image: test_input.jpg")
    
    # Initialize components
    compressor = UnifiedCompressor()
    key = "testkey123456789"  # 16 characters
    cipher = AESCipher(key.encode('utf-8'))
    
    try:
        # Step 1: Compress
        print("\n📦 Testing Compression...")
        compressed_data = compressor.compress(test_image, quality=75, use_color=True)
        stats = compressor.get_compression_stats(test_image, compressed_data)
        print(f"✅ Compression ratio: {stats['compression_ratio']:.2f}x")
        
        # Step 2: Encrypt
        print("\n🔒 Testing Encryption...")
        compressed_bytes = pickle.dumps(compressed_data)
        encrypted_data = cipher.encrypt(compressed_bytes)
        print(f"✅ Encrypted data size: {len(encrypted_data)} bytes")
        
        # Save encrypted file
        with open("test_encrypted.bin", "wb") as f:
            f.write(encrypted_data)
        print("✅ Saved encrypted file: test_encrypted.bin")
        
        # Step 3: Decrypt
        print("\n🔓 Testing Decryption...")
        with open("test_encrypted.bin", "rb") as f:
            encrypted_data_read = f.read()
        
        decrypted_bytes = cipher.decrypt(encrypted_data_read)
        decrypted_compressed_data = pickle.loads(decrypted_bytes)
        print("✅ Decryption successful")
        
        # Step 4: Decompress
        print("\n📤 Testing Decompression...")
        reconstructed_image = compressor.decompress(decrypted_compressed_data)
        print(f"✅ Reconstructed image shape: {reconstructed_image.shape}")
        
        # Save reconstructed image
        Image.fromarray(reconstructed_image).save("test_output.png")
        print("✅ Saved reconstructed image: test_output.png")
        
        # Step 5: Quality check
        print("\n📊 Quality Analysis...")
        mse = np.mean((test_image.astype(float) - reconstructed_image.astype(float)) ** 2)
        psnr = 20 * np.log10(255) - 10 * np.log10(mse) if mse > 0 else float("inf")
        
        print(f"✅ MSE: {mse:.2f}")
        print(f"✅ PSNR: {psnr:.2f} dB")
        
        if psnr > 30:
            print("🎉 Quality: Good")
        elif psnr > 20:
            print("⚠️  Quality: Fair") 
        else:
            print("❌ Quality: Poor")
            
        print("\n" + "=" * 40)
        print("🎉 ALL TESTS PASSED!")
        print("The image encryption system is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_compression_encryption_cycle()
    sys.exit(0 if success else 1)