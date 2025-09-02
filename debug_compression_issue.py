#!/usr/bin/env python3
"""
Debug the compression size issue - 300KB → 60MB problem.
Test the full API workflow to identify where data expansion occurs.
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import time

# Add api to path
sys.path.append('api')
sys.path.append('.')

from api.dct2dff_compressor import DCT2DFFCompressor
from aes_cipher import AESCipher

def analyze_sizes(data, label):
    """Analyze data size at each step."""
    if isinstance(data, np.ndarray):
        size = data.nbytes
        print(f"  {label}: {size:,} bytes ({size/1024:.1f} KB) - numpy array {data.shape}")
    elif isinstance(data, bytes):
        size = len(data)
        print(f"  {label}: {size:,} bytes ({size/1024:.1f} KB) - bytes")
    elif isinstance(data, dict):
        pickled_size = len(pickle.dumps(data))
        print(f"  {label}: {pickled_size:,} bytes ({pickled_size/1024:.1f} KB) - dict (pickled)")
    else:
        try:
            size = len(str(data).encode())
            print(f"  {label}: {size:,} bytes ({size/1024:.1f} KB) - other")
        except:
            print(f"  {label}: Unknown size")

def test_compression_pipeline():
    """Test the full compression pipeline step by step."""
    print("🔍 DEBUGGING COMPRESSION ISSUE")
    print("=" * 60)
    
    # Load a real test image
    test_paths = ['sample_image.jpg', 'test_input.jpg', 'api/test_input.jpg', 'api/test_input.png']
    
    test_image = None
    image_path = None
    
    for path in test_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path)
                if img.mode in ['RGB', 'L']:
                    test_image = np.array(img)
                    image_path = path
                    break
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    if test_image is None:
        # Create a test image
        print("📸 Creating synthetic test image...")
        test_image = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        image_path = "synthetic"
    else:
        print(f"📸 Using real image: {image_path}")
    
    print(f"Image shape: {test_image.shape}")
    
    # Step 1: Original image
    print(f"\n1️⃣ ORIGINAL IMAGE")
    analyze_sizes(test_image, "Original image")
    
    # Step 2: DCT Compression
    print(f"\n2️⃣ DCT COMPRESSION")
    compressor = DCT2DFFCompressor()
    
    # Test different quality levels
    for quality in [10, 25, 50, 75]:
        print(f"\n  Quality {quality}%:")
        compressed_data = compressor.compress(test_image, quality=quality)
        
        analyze_sizes(compressed_data, f"Compressed data (Q{quality})")
        
        # Check the coefficient data specifically
        if 'dct_coefficients' in compressed_data:
            coeffs = compressed_data['dct_coefficients']
            if isinstance(coeffs, list) and len(coeffs) > 0:
                if isinstance(coeffs[0], list):  # Color image
                    total_blocks = sum(len(channel) for channel in coeffs)
                    print(f"    Color channels: 3, Total blocks: {total_blocks}")
                    # Sample first few blocks
                    sample_block = coeffs[0][0] if coeffs[0] else None
                else:  # Grayscale
                    total_blocks = len(coeffs)
                    print(f"    Grayscale, Total blocks: {total_blocks}")
                    sample_block = coeffs[0] if coeffs else None
                
                if sample_block:
                    print(f"    Sample block keys: {list(sample_block.keys())}")
                    print(f"    Sample coeffs size: {len(sample_block.get('coefficients', []))}")
                    print(f"    Sample indices size: {len(sample_block.get('indices', []))}")
    
    # Step 3: Use quality 75 for detailed analysis
    print(f"\n3️⃣ DETAILED ANALYSIS (Quality 75%)")
    compressed_data = compressor.compress(test_image, quality=75)
    
    # Calculate compression ratio
    original_size = test_image.nbytes
    compressed_pickled = pickle.dumps(compressed_data)
    compressed_size = len(compressed_pickled)
    ratio = compressed_size / original_size
    
    print(f"  Compression ratio: {ratio:.2f} ({ratio*100:.1f}%)")
    if ratio > 1.0:
        print(f"  ❌ DATA EXPANSION: {(ratio-1)*100:.1f}% larger!")
    else:
        print(f"  ✅ Compression: {(1-ratio)*100:.1f}% space saved")
    
    # Step 4: Test decompression
    print(f"\n4️⃣ DECOMPRESSION TEST")
    try:
        reconstructed = compressor.decompress(compressed_data)
        analyze_sizes(reconstructed, "Reconstructed image")
        
        # Calculate PSNR
        psnr = compressor.calculate_psnr(test_image, reconstructed)
        print(f"  PSNR: {psnr:.2f} dB")
        
        # Check if images match in shape
        if test_image.shape == reconstructed.shape:
            print(f"  ✅ Shape preserved: {test_image.shape}")
        else:
            print(f"  ❌ Shape mismatch: {test_image.shape} → {reconstructed.shape}")
            
    except Exception as e:
        print(f"  ❌ Decompression failed: {e}")
    
    # Step 5: Full encryption pipeline
    print(f"\n5️⃣ FULL ENCRYPTION PIPELINE")
    key = "testkey123456789"  # 16 chars
    cipher = AESCipher(key.encode())
    
    # Serialize compressed data
    serialized = pickle.dumps(compressed_data)
    analyze_sizes(serialized, "Serialized (pre-encryption)")
    
    # Encrypt
    encrypted = cipher.encrypt(serialized)
    analyze_sizes(encrypted, "Encrypted data")
    
    # Test decryption
    try:
        decrypted_bytes = cipher.decrypt(encrypted)
        analyze_sizes(decrypted_bytes, "Decrypted bytes")
        
        deserialized = pickle.loads(decrypted_bytes)
        final_image = compressor.decompress(deserialized)
        analyze_sizes(final_image, "Final reconstructed")
        
        # Final ratio
        final_ratio = len(encrypted) / original_size
        print(f"\n📊 FINAL ANALYSIS:")
        print(f"  Original: {original_size:,} bytes ({original_size/1024:.1f} KB)")
        print(f"  Encrypted: {len(encrypted):,} bytes ({len(encrypted)/1024:.1f} KB)")
        print(f"  Final ratio: {final_ratio:.2f} ({final_ratio*100:.1f}%)")
        
        if final_ratio > 20:  # More than 20x expansion is clearly wrong
            print(f"  ❌ CRITICAL: {final_ratio:.1f}x expansion detected!")
            print(f"  🔍 Root cause analysis needed")
        
    except Exception as e:
        print(f"  ❌ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()

def analyze_coefficient_storage():
    """Analyze why coefficient storage is so large."""
    print(f"\n🔬 COEFFICIENT STORAGE ANALYSIS")
    print("=" * 50)
    
    # Create small test case
    test_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    compressor = DCT2DFFCompressor()
    
    compressed_data = compressor.compress(test_image, quality=50)
    
    # Analyze the storage format
    coeffs = compressed_data['dct_coefficients']
    
    print(f"Image: {test_image.shape} = {test_image.size} pixels")
    print(f"Blocks needed: {(64//8) * (64//8) * 3} = {(64//8) * (64//8) * 3} total")
    
    total_stored_data = 0
    
    for ch_idx, channel_coeffs in enumerate(coeffs):
        channel_size = 0
        for block in channel_coeffs:
            # Calculate actual storage per block
            block_size = (
                len(pickle.dumps(block['position'])) +
                len(pickle.dumps(block['indices'])) +
                len(pickle.dumps(block['coefficients']))
            )
            channel_size += block_size
        
        total_stored_data += channel_size
        print(f"Channel {ch_idx}: {len(channel_coeffs)} blocks, {channel_size:,} bytes")
    
    print(f"Total coefficient data: {total_stored_data:,} bytes")
    print(f"Original image: {test_image.nbytes:,} bytes")
    print(f"Storage ratio: {total_stored_data / test_image.nbytes:.2f}")

if __name__ == "__main__":
    test_compression_pipeline()
    analyze_coefficient_storage()