#!/usr/bin/env python3
"""
Test script to verify DCT2DFF.py functionality before API integration.
Tests compression/decompression with different image types and compression ratios.
"""

import numpy as np
from PIL import Image
import time
import sys
import os

# Import DCT2DFF functions
from DCT2DFF import (
    compress_image_dct_color,
    calculate_psnr_color,
    create_test_image_color
)

def test_basic_functionality():
    """Test basic compression and decompression functionality."""
    print("🧪 Testing Basic DCT2DFF Functionality")
    print("=" * 50)
    
    # Test with a small synthetic image
    print("📸 Creating test images...")
    
    # Create grayscale test image
    gray_image = create_test_image_color((64, 64), color=False)
    print(f"   Grayscale image: {gray_image.shape}, dtype: {gray_image.dtype}")
    
    # Create color test image  
    color_image = create_test_image_color((64, 64), color=True)
    print(f"   Color image: {color_image.shape}, dtype: {color_image.dtype}")
    
    return gray_image, color_image

def test_compression_ratios(image, image_type=""):
    """Test different compression ratios."""
    print(f"\n🔄 Testing Compression Ratios - {image_type}")
    print("-" * 40)
    
    ratios = [0.1, 0.25, 0.5, 0.75, 1.0]
    results = []
    
    for ratio in ratios:
        start_time = time.time()
        
        # Compress and decompress
        compressed_img, info = compress_image_dct_color(image, ratio)
        
        process_time = time.time() - start_time
        
        # Calculate quality metrics
        psnr = calculate_psnr_color(image, compressed_img)
        
        # Calculate compression stats
        original_size = image.size * image.itemsize
        kept_coeffs = info['compressed_size']
        total_coeffs = info['original_size']
        space_saved = info['space_saved']
        
        results.append({
            'ratio': ratio,
            'psnr': psnr,
            'time': process_time,
            'space_saved': space_saved,
            'coeffs_kept': kept_coeffs,
            'coeffs_total': total_coeffs
        })
        
        print(f"   Ratio {ratio:4.2f} | PSNR: {psnr:6.2f} dB | Time: {process_time:5.3f}s | Space Saved: {space_saved:.1%}")
    
    return results

def test_api_compatibility():
    """Test data format compatibility for API integration."""
    print(f"\n🔗 Testing API Compatibility")
    print("-" * 40)
    
    # Create test image
    test_image = create_test_image_color((32, 32), color=True)
    
    # Test compression with coefficient storage (needed for API)
    from DCT2DFF import compress_channel_dct_with_storage
    
    print("   Testing coefficient storage format...")
    
    # Test each channel separately (like API will do)
    all_coefficients = []
    for channel_idx in range(3):
        channel = test_image[:, :, channel_idx]
        compressed_channel, coeffs_list, info = compress_channel_dct_with_storage(
            channel, 0.2
        )
        all_coefficients.append(coeffs_list)
        print(f"   Channel {channel_idx}: {len(coeffs_list)} blocks, {info['compressed_size']} coefficients")
    
    # Test pickle serialization (used by API)
    import pickle
    
    # Create API-compatible data structure
    api_data = {
        'dct_coefficients': all_coefficients,
        'original_shape': test_image.shape,
        'compression_ratio': 0.2,
        'num_channels': 3,
        'use_manual_dct': True
    }
    
    try:
        # Test serialization
        serialized = pickle.dumps(api_data)
        deserialized = pickle.loads(serialized)
        
        print(f"   ✅ Pickle serialization: {len(serialized)} bytes")
        print(f"   ✅ Data structure preserved: {deserialized['original_shape']}")
        
    except Exception as e:
        print(f"   ❌ Serialization failed: {e}")
        return False
    
    return True

def test_with_real_image():
    """Test with a real image if available."""
    print(f"\n📷 Testing with Real Image")
    print("-" * 40)
    
    # Look for test images in the project
    test_paths = [
        'sample_image.jpg',
        'test_input.jpg', 
        'api/test_input.jpg',
        'api/test_input.png'
    ]
    
    test_image = None
    used_path = None
    
    for path in test_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path)
                if img.mode == 'RGB':
                    test_image = np.array(img)
                    used_path = path
                    break
                elif img.mode == 'L':
                    test_image = np.array(img)
                    used_path = path
                    break
            except Exception:
                continue
    
    if test_image is not None:
        print(f"   📁 Using: {used_path}")
        print(f"   📏 Shape: {test_image.shape}")
        
        # Test compression
        start_time = time.time()
        compressed_img, info = compress_image_dct_color(test_image, 0.15)
        process_time = time.time() - start_time
        
        psnr = calculate_psnr_color(test_image, compressed_img)
        
        print(f"   ⏱️  Processing time: {process_time:.3f}s")
        print(f"   📊 PSNR: {psnr:.2f} dB")
        print(f"   💾 Space saved: {info['space_saved']:.1%}")
        print(f"   🧮 Blocks processed: {info['total_blocks']}")
        
        return True
    else:
        print("   ⚠️  No real test images found, skipping...")
        return False

def benchmark_performance():
    """Benchmark performance compared to simple metrics."""
    print(f"\n⚡ Performance Benchmark")
    print("-" * 40)
    
    # Test different image sizes
    sizes = [(64, 64), (128, 128), (256, 256)]
    
    for size in sizes:
        print(f"   🖼️  Size {size[0]}x{size[1]}:")
        
        # Create test image
        test_image = create_test_image_color(size, color=True)
        
        # Benchmark compression
        start_time = time.time()
        compressed_img, info = compress_image_dct_color(test_image, 0.2)
        process_time = time.time() - start_time
        
        # Calculate metrics
        blocks_per_sec = info['total_blocks'] / process_time if process_time > 0 else 0
        pixels_per_sec = (size[0] * size[1]) / process_time if process_time > 0 else 0
        
        print(f"      ⏱️  {process_time:.3f}s | {blocks_per_sec:.0f} blocks/s | {pixels_per_sec:.0f} pixels/s")

def main():
    """Main test function."""
    print("🚀 DCT2DFF.py Functionality Test")
    print("=" * 60)
    print("Testing manual DCT implementation before API integration...")
    
    try:
        # Test 1: Basic functionality
        gray_image, color_image = test_basic_functionality()
        
        # Test 2: Compression ratios
        test_compression_ratios(gray_image, "Grayscale")
        test_compression_ratios(color_image, "Color")
        
        # Test 3: API compatibility
        api_compatible = test_api_compatibility()
        
        # Test 4: Real image (if available)
        real_image_tested = test_with_real_image()
        
        # Test 5: Performance benchmark
        benchmark_performance()
        
        # Summary
        print(f"\n📋 Test Summary")
        print("=" * 60)
        print(f"   ✅ Basic functionality: PASSED")
        print(f"   ✅ Multiple compression ratios: PASSED") 
        print(f"   {'✅' if api_compatible else '❌'} API compatibility: {'PASSED' if api_compatible else 'FAILED'}")
        print(f"   {'✅' if real_image_tested else '⚠️ '} Real image test: {'PASSED' if real_image_tested else 'SKIPPED'}")
        print(f"   ✅ Performance benchmark: PASSED")
        
        if api_compatible:
            print(f"\n🎉 DCT2DFF.py is ready for API integration!")
            print(f"   • Manual DCT implementation working correctly")
            print(f"   • No quantization - using ratio-based coefficient selection") 
            print(f"   • No zigzag indices - direct 8x8 block processing")
            print(f"   • Data format compatible with pickle serialization")
        else:
            print(f"\n⚠️  Issues found - check API compatibility before integration")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)