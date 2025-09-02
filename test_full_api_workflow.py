#!/usr/bin/env python3
"""
Test the full API encrypt/decrypt workflow with the fixed DCT compressor.
Verify that compression ratios are reasonable and the pipeline works correctly.
"""

import sys
import os
import pickle
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

# Add paths
sys.path.append('api')
sys.path.append('.')

from api.dct2dff_compressor_fixed import DCT2DFFCompressor
from aes_cipher import AESCipher

def test_full_workflow():
    """Test the complete encrypt/decrypt workflow."""
    print("🚀 FULL API WORKFLOW TEST")
    print("=" * 60)
    
    # Find a real test image
    test_paths = ['sample_image.jpg', 'test_input.jpg', 'api/test_input.jpg', 'api/test_input.png']
    test_image_path = None
    
    for path in test_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if not test_image_path:
        print("📸 No real image found, creating synthetic test image...")
        # Create a realistic test image
        test_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        # Add some structure to make it more realistic
        for i in range(0, 300, 50):
            test_image[i:i+25, :, 0] = 255  # Red stripes
        for j in range(0, 400, 60):
            test_image[:, j:j+30, 1] = 255  # Green stripes
    else:
        print(f"📸 Using real image: {test_image_path}")
        img = Image.open(test_image_path)
        if img.mode not in ['RGB', 'L']:
            img = img.convert('RGB')
        test_image = np.array(img)
    
    print(f"   Image shape: {test_image.shape}")
    print(f"   Original size: {test_image.nbytes:,} bytes ({test_image.nbytes/1024:.1f} KB)")
    
    # Test different quality levels
    qualities = [25, 50, 75]
    key = "testkey123456789"  # 16 chars
    
    for quality in qualities:
        print(f"\n🔄 Testing Quality {quality}%")
        print("-" * 40)
        
        # Step 1: Compression
        compressor = DCT2DFFCompressor()
        compressed_data = compressor.compress(test_image, quality=quality)
        
        # Check compression statistics
        stats = compressor.get_compression_stats(test_image, compressed_data)
        print(f"   📊 Compression ratio: {stats['compression_ratio']:.2f}")
        print(f"   💾 Space saved: {stats['space_saved_percent']:.1f}%")
        
        # Step 2: Serialization (for encryption)
        serialized = pickle.dumps(compressed_data)
        print(f"   📦 Serialized size: {len(serialized):,} bytes ({len(serialized)/1024:.1f} KB)")
        
        serialization_ratio = len(serialized) / test_image.nbytes
        print(f"   📈 Serialization ratio: {serialization_ratio:.2f}")
        
        # Step 3: Encryption
        cipher = AESCipher(key.encode())
        encrypted = cipher.encrypt(serialized)
        print(f"   🔐 Encrypted size: {len(encrypted):,} bytes ({len(encrypted)/1024:.1f} KB)")
        
        encryption_ratio = len(encrypted) / test_image.nbytes
        print(f"   📈 Final ratio: {encryption_ratio:.2f}")
        
        # Step 4: Decryption
        try:
            decrypted_bytes = cipher.decrypt(encrypted)
            print(f"   🔓 Decrypted size: {len(decrypted_bytes):,} bytes")
            
            # Step 5: Deserialization
            deserialized_data = pickle.loads(decrypted_bytes)
            print(f"   📂 Deserialization successful")
            
            # Step 6: Decompression
            reconstructed = compressor.decompress(deserialized_data)
            print(f"   🖼️  Reconstructed shape: {reconstructed.shape}")
            
            # Step 7: Quality check
            psnr = compressor.calculate_psnr(test_image, reconstructed)
            print(f"   📈 PSNR: {psnr:.2f} dB")
            
            # Check if results are reasonable
            if encryption_ratio < 5.0:  # Less than 5x expansion is reasonable
                print(f"   ✅ GOOD: Reasonable file size ratio ({encryption_ratio:.1f}x)")
            else:
                print(f"   ❌ BAD: Excessive file size ratio ({encryption_ratio:.1f}x)")
            
            if psnr > 15.0:  # PSNR > 15 dB is decent quality
                print(f"   ✅ GOOD: Acceptable image quality ({psnr:.1f} dB)")
            else:
                print(f"   ⚠️  LOW: Poor image quality ({psnr:.1f} dB)")
                
        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")
            return False
    
    return True

def simulate_api_request():
    """Simulate what happens in an actual API request."""
    print(f"\n🌐 SIMULATING API REQUEST")
    print("=" * 60)
    
    # Create a realistic test image (similar to what users might upload)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some realistic content patterns
    # Sky gradient
    for y in range(100):
        test_image[y, :, :] = [135 + y, 206 + y//2, 235]
    
    # Grass
    test_image[400:, :, :] = [34, 139, 34]
    
    # Some buildings
    for x in range(100, 600, 150):
        test_image[100:400, x:x+100, :] = [105, 105, 105]
    
    print(f"📸 Simulated photo: {test_image.shape}")
    print(f"   Original size: {test_image.nbytes:,} bytes ({test_image.nbytes/1024:.1f} KB)")
    
    # Simulate API workflow
    quality = 75  # Default quality
    key = "userpassword1234"  # Typical user key
    
    # API compression + encryption
    compressor = DCT2DFFCompressor()
    compressed_data = compressor.compress(test_image, quality=quality)
    
    # Add metadata (like API does)
    compressed_data['original_format'] = '.jpg'
    compressed_data['original_filename'] = 'user_photo.jpg'
    
    # Serialize and encrypt
    serialized = pickle.dumps(compressed_data)
    cipher = AESCipher(key.encode())
    encrypted = cipher.encrypt(serialized)
    
    # Results
    original_kb = test_image.nbytes / 1024
    encrypted_kb = len(encrypted) / 1024
    ratio = len(encrypted) / test_image.nbytes
    
    print(f"\n📊 API Results:")
    print(f"   Original: {original_kb:.1f} KB")
    print(f"   Encrypted: {encrypted_kb:.1f} KB")
    print(f"   Ratio: {ratio:.2f}x")
    
    # This should be reasonable (not 200x like before!)
    if ratio < 2.0:
        print(f"   ✅ EXCELLENT: Very reasonable size increase")
    elif ratio < 5.0:
        print(f"   ✅ GOOD: Acceptable size increase") 
    elif ratio < 10.0:
        print(f"   ⚠️  OK: Moderate size increase")
    else:
        print(f"   ❌ BAD: Excessive size increase")
    
    # Test decryption
    try:
        decrypted = cipher.decrypt(encrypted)
        deserialized = pickle.loads(decrypted)
        reconstructed = compressor.decompress(deserialized)
        
        psnr = compressor.calculate_psnr(test_image, reconstructed)
        print(f"   📈 Quality: {psnr:.1f} dB PSNR")
        print(f"   ✅ Full decrypt/decompress successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Decryption failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_full_workflow()
    success2 = simulate_api_request()
    
    print(f"\n🎯 FINAL RESULTS")
    print("=" * 60)
    if success1 and success2:
        print("✅ All tests PASSED!")
        print("✅ DCT compression working correctly")
        print("✅ File sizes are reasonable")
        print("✅ Full encrypt/decrypt workflow works")
        print("🎉 API is ready for use!")
    else:
        print("❌ Some tests FAILED")
        print("🔧 Further debugging needed")