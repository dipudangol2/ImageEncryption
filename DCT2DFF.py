import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

# Pre-compute DCT and IDCT matrices for 8x8 blocks (done once for efficiency)
def create_dct_matrices():
    """
    Pre-compute DCT and IDCT transformation matrices for 8x8 blocks
    This is done once and reused for all blocks, making it much faster
    """
    N = 8
    dct_matrix = np.zeros((N, N))
    
    for u in range(N):
        for x in range(N):
            if u == 0:
                dct_matrix[u, x] = np.sqrt(1/N)
            else:
                dct_matrix[u, x] = np.sqrt(2/N) * np.cos((2*x + 1) * u * np.pi / (2*N))
    
    # IDCT matrix is the transpose of DCT matrix
    idct_matrix = dct_matrix.T
    
    return dct_matrix, idct_matrix

# Global matrices (computed once)
DCT_MATRIX, IDCT_MATRIX = create_dct_matrices()

def dct2d_fast(block):
    """
    Fast 2D DCT using pre-computed matrices and matrix multiplication
    
    Args:
        block: 8x8 numpy array
    
    Returns:
        8x8 DCT coefficients
    """
    # 2D DCT = DCT_MATRIX @ block @ DCT_MATRIX.T
    return DCT_MATRIX @ block @ DCT_MATRIX.T

def idct2d_fast(dct_block):
    """
    Fast 2D Inverse DCT using pre-computed matrices
    
    Args:
        dct_block: 8x8 DCT coefficients
    
    Returns:
        8x8 reconstructed block
    """
    # 2D IDCT = IDCT_MATRIX @ dct_block @ IDCT_MATRIX.T
    return IDCT_MATRIX @ dct_block @ IDCT_MATRIX.T

def compress_channel_dct_with_storage(channel, compression_ratio):
    """
    DCT compression that returns both reconstructed image and compressed coefficients
    
    Args:
        channel: Input grayscale channel as numpy array
        compression_ratio: Float between 0 and 1, percentage of coefficients to keep
    
    Returns:
        compressed_channel: Reconstructed channel after compression
        compressed_coefficients: List of (position, coefficients) for storage
        compression_info: Dictionary with compression statistics
    """
    start_time = time.time()
    
    # Ensure image dimensions are multiples of 8
    h, w = channel.shape
    h_pad = (8 - h % 8) % 8
    w_pad = (8 - w % 8) % 8
    
    if h_pad > 0 or w_pad > 0:
        channel = np.pad(channel, ((0, h_pad), (0, w_pad)), mode='edge')
    
    new_h, new_w = channel.shape
    
    # Convert to float and center around 0
    channel_float = channel.astype(np.float32) - 128
    
    # Create output array and storage for compressed coefficients
    compressed_channel = np.zeros_like(channel_float)
    compressed_coefficients = []
    
    # Number of coefficients to keep per block
    keep_coeffs = max(1, int(64 * compression_ratio))
    total_blocks = 0
    
    # Process image in 8x8 blocks
    for i in range(0, new_h, 8):
        for j in range(0, new_w, 8):
            # Extract 8x8 block
            block = channel_float[i:i+8, j:j+8]
            
            # Apply fast DCT
            dct_coeffs = dct2d_fast(block)
            
            # Compress by keeping only the largest coefficients
            if compression_ratio < 1.0:
                # Flatten coefficients and find largest ones
                flat_coeffs = dct_coeffs.flatten()
                
                # Use argpartition for efficient partial sorting
                indices = np.argpartition(np.abs(flat_coeffs), -keep_coeffs)[-keep_coeffs:]
                
                # Store only the non-zero coefficients and their positions
                non_zero_coeffs = flat_coeffs[indices]
                compressed_coefficients.append({
                    'position': (i, j),
                    'indices': indices,
                    'coefficients': non_zero_coeffs
                })
                
                # Create compressed version for reconstruction
                compressed_flat = np.zeros_like(flat_coeffs)
                compressed_flat[indices] = flat_coeffs[indices]
                compressed_dct = compressed_flat.reshape(8, 8)
            else:
                compressed_dct = dct_coeffs
                compressed_coefficients.append({
                    'position': (i, j),
                    'indices': np.arange(64),
                    'coefficients': dct_coeffs.flatten()
                })
            
            # Apply inverse DCT
            reconstructed_block = idct2d_fast(compressed_dct)
            
            # Place back in image
            compressed_channel[i:i+8, j:j+8] = reconstructed_block
            total_blocks += 1
    
    # Shift back to [0, 255] range and clip
    compressed_channel = compressed_channel + 128
    compressed_channel = np.clip(compressed_channel, 0, 255)
    
    # Remove padding if it was added
    if h_pad > 0 or w_pad > 0:
        compressed_channel = compressed_channel[:h, :w]
    
    processing_time = time.time() - start_time
    
    compression_info = {
        'original_size': total_blocks * 64,
        'compressed_size': total_blocks * keep_coeffs,
        'compression_ratio': compression_ratio,
        'space_saved': 1 - (keep_coeffs / 64),
        'total_blocks': total_blocks,
        'processing_time': processing_time,
        'actual_coefficients_stored': len(compressed_coefficients) * keep_coeffs
    }
    
    return compressed_channel.astype(np.uint8), compressed_coefficients, compression_info

def compress_channel_dct_optimized(channel, compression_ratio):
    """
    Optimized DCT compression for single channel (without coefficient storage)
    
    Args:
        channel: Input grayscale channel as numpy array
        compression_ratio: Float between 0 and 1, percentage of coefficients to keep
    
    Returns:
        compressed_channel: Reconstructed channel after compression
        compression_info: Dictionary with compression statistics
    """
    compressed_channel, _, compression_info = compress_channel_dct_with_storage(channel, compression_ratio)
    return compressed_channel, compression_info

def compress_image_dct_color(image, compression_ratio):
    """
    DCT compression that preserves color by processing each channel separately
    
    Args:
        image: Input color image as numpy array (H x W x 3) or grayscale (H x W)
        compression_ratio: Float between 0 and 1, percentage of coefficients to keep
    
    Returns:
        compressed_image: Reconstructed image after compression
        compression_info: Dictionary with compression statistics
    """
    start_time = time.time()
    
    # Handle both color and grayscale images
    if len(image.shape) == 2:
        # Grayscale image
        compressed_channel, info = compress_channel_dct_optimized(image, compression_ratio)
        return compressed_channel, info
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Color image - process each channel separately
        compressed_channels = []
        total_info = None
        
        channel_names = ['Red', 'Green', 'Blue']
        
        for channel_idx in range(3):
            print(f"Processing {channel_names[channel_idx]} channel...")
            channel = image[:, :, channel_idx]
            compressed_channel, channel_info = compress_channel_dct_optimized(channel, compression_ratio)
            compressed_channels.append(compressed_channel)
            
            # Accumulate timing information
            if total_info is None:
                total_info = channel_info.copy()
            else:
                total_info['processing_time'] += channel_info['processing_time']
                total_info['original_size'] += channel_info['original_size']
                total_info['compressed_size'] += channel_info['compressed_size']
        
        # Combine channels back into color image
        compressed_image = np.stack(compressed_channels, axis=2)
        
        # Update total processing time
        total_info['processing_time'] = time.time() - start_time
        total_info['total_blocks'] *= 3  # 3 channels
        
        return compressed_image, total_info
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}. Expected (H, W) or (H, W, 3)")

def calculate_psnr_color(original, compressed):
    """Calculate Peak Signal-to-Noise Ratio for color images"""
    if len(original.shape) == 2:
        # Grayscale
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    else:
        # Color - calculate MSE across all channels
        mse = np.mean((original.astype(np.float32) - compressed.astype(np.float32)) ** 2)
    
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def save_compressed_data(dct_coefficients, image_shape, compression_ratio, output_dir="dct_output"):
    """
    Save the actual compressed DCT coefficients (the real compressed data)
    
    Args:
        dct_coefficients: List of compressed DCT coefficient arrays
        image_shape: Original image shape
        compression_ratio: Compression ratio used
        output_dir: Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save compressed coefficients as numpy file
    coeffs_path = os.path.join(output_dir, f"dct_coeffs_{compression_ratio:.3f}_{timestamp}.npz")
    np.savez_compressed(coeffs_path, 
                       coefficients=dct_coefficients, 
                       shape=image_shape,
                       ratio=compression_ratio)
    
    # Calculate actual file sizes
    original_size = np.prod(image_shape) if len(image_shape) == 2 else np.prod(image_shape)
    compressed_file_size = os.path.getsize(coeffs_path)
    
    return {
        'coeffs_path': coeffs_path,
        'original_data_size': original_size,  # in pixels
        'compressed_file_size': compressed_file_size,  # in bytes
        'actual_compression_ratio': compressed_file_size / (original_size * (3 if len(image_shape) == 3 else 1))
    }

def save_images_color(original, compressed, compression_ratio, output_dir="dct_output"):
    """
    Save original and compressed images to files with file size analysis
    
    Args:
        original: Original image array
        compressed: Compressed image array
        compression_ratio: Compression ratio used
        output_dir: Directory to save images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Determine if image is color or grayscale
    is_color = len(original.shape) == 3 and original.shape[2] == 3
    suffix = "color" if is_color else "gray"
    
    # Save original image
    original_path = os.path.join(output_dir, f"original_{suffix}_{timestamp}.png")
    if is_color:
        Image.fromarray(original).save(original_path)
    else:
        Image.fromarray(original, mode='L').save(original_path)
    
    # Save compressed image
    ratio_str = f"{compression_ratio:.3f}"
    compressed_path = os.path.join(output_dir, f"compressed_{suffix}_{ratio_str}_{timestamp}.jpg")
    if is_color:
        Image.fromarray(compressed).save(compressed_path)
    else:
        Image.fromarray(compressed, mode='L').save(compressed_path)
    
    # Save difference image
    diff = np.abs(original.astype(np.float32) - compressed.astype(np.float32))
    if is_color:
        # For color images, show the magnitude of difference across all channels
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))
        diff_normalized = (diff_magnitude / diff_magnitude.max() * 255).astype(np.uint8)
        diff_path = os.path.join(output_dir, f"difference_{suffix}_{ratio_str}_{timestamp}.png")
        Image.fromarray(diff_normalized, mode='L').save(diff_path)
    else:
        diff_normalized = (diff / diff.max() * 255).astype(np.uint8)
        diff_path = os.path.join(output_dir, f"difference_{suffix}_{ratio_str}_{timestamp}.png")
        Image.fromarray(diff_normalized, mode='L').save(diff_path)
    
    # Calculate and display file sizes
    original_file_size = os.path.getsize(original_path)
    compressed_file_size = os.path.getsize(compressed_path)
    
    print(f"\nFile Size Analysis:")
    print(f"Original PNG file: {original_file_size:,} bytes ({original_file_size/1024:.1f} KB)")
    print(f"Compressed PNG file: {compressed_file_size:,} bytes ({compressed_file_size/1024:.1f} KB)")
    print(f"PNG file size ratio: {compressed_file_size/original_file_size:.3f}")
    print(f"Note: PNG compression is separate from DCT compression!")
    
    return {
        'original_path': original_path,
        'compressed_path': compressed_path,
        'difference_path': diff_path,
        'original_file_size': original_file_size,
        'compressed_file_size': compressed_file_size
    }

def create_test_image_color(size=(512, 512), color=True):
    """Create a test image with various patterns for testing (color or grayscale)"""
    h, w = size
    
    if not color:
        # Return grayscale test image
        img = np.zeros((h, w), dtype=np.uint8)
        
        # Create multiple patterns for better testing
        # Checkerboard pattern
        for i in range(0, h, 32):
            for j in range(0, w, 32):
                if (i//32 + j//32) % 2:
                    img[i:min(i+32, h), j:min(j+32, w)] = 255
        
        # Add circular gradient
        center_h, center_w = h//2, w//2
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        max_distance = np.sqrt((h/2)**2 + (w/2)**2)
        
        # Create gradient overlay
        gradient = (1 - distance / max_distance) * 100
        img = np.clip(img.astype(np.float32) + gradient, 0, 255).astype(np.uint8)
        
        return img
    
    # Create color test image
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Create different patterns for each channel
    center_h, center_w = h//2, w//2
    y, x = np.ogrid[:h, :w]
    
    # Red channel - radial gradient
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    max_distance = np.sqrt((h/2)**2 + (w/2)**2)
    img[:, :, 0] = (1 - distance / max_distance) * 255
    
    # Green channel - horizontal gradient with sine wave
    img[:, :, 1] = (x / w * 255 + 50 * np.sin(y / 20)).clip(0, 255)
    
    # Blue channel - checkerboard with vertical gradient
    for i in range(0, h, 64):
        for j in range(0, w, 64):
            if (i//64 + j//64) % 2:
                img[i:min(i+64, h), j:min(j+64, w), 2] = 255
    
    # Add vertical gradient to blue channel
    img[:, :, 2] = (img[:, :, 2] * 0.7 + (y / h * 255) * 0.3).clip(0, 255)
    
    # Add some high-frequency details
    for i in range(50, h-50, 150):
        for j in range(50, w-50, 150):
            # Colorful rectangles
            img[i:i+20, j:j+40, 0] = 255  # Red rectangle
            img[i+30:i+50, j:j+40, 1] = 255  # Green rectangle
            img[i+60:i+80, j:j+40, 2] = 255  # Blue rectangle
    
    return img

def load_or_create_image_color():
    """Load an image or create a test image (supports color)"""
    print("\nImage Source Options:")
    print("1. Generated color test image (512x512)")
    print("2. Generated grayscale test image (512x512)")
    print("3. Generated large color test image (1024x1024)")
    print("4. Load your own image")
    
    choice = input("Enter choice (1-4, default 1): ").strip()
    
    if choice == "2":
        print("Creating grayscale test image (512x512)...")
        return create_test_image_color((512, 512), color=False)
    elif choice == "3":
        print("Creating large color test image (1024x1024)...")
        return create_test_image_color((1024, 1024), color=True)
    elif choice == "4":
        image_path = input("Enter image path: ").strip()
        if image_path and os.path.exists(image_path):
            try:
                img_pil = Image.open(image_path)
                # Convert to RGB if it's not already
                if img_pil.mode != 'RGB' and img_pil.mode != 'L':
                    img_pil = img_pil.convert('RGB')
                
                image = np.array(img_pil)
                print(f"Loaded image: {image_path}, shape: {image.shape}, mode: {img_pil.mode}")
                return image
            except Exception as e:
                print(f"Error loading image: {e}")
                print("Using default color test image instead...")
                return create_test_image_color()
        else:
            print("Invalid path. Using default color test image...")
            return create_test_image_color()
    else:
        print("Creating color test image (512x512)...")
        return create_test_image_color()

def benchmark_compression(image, compression_ratio=0.1):
    """Benchmark compression implementation"""
    print(f"\nBenchmarking on {image.shape} image:")
    print("-" * 60)
    
    start_time = time.time()
    compressed_img, info = compress_image_dct_color(image, compression_ratio)
    process_time = time.time() - start_time
    psnr = calculate_psnr_color(image, compressed_img)
    
    total_blocks = info['total_blocks']
    if len(image.shape) == 3:
        print(f"RGB compression:   {process_time:.3f}s | PSNR: {psnr:.2f} dB | {total_blocks/process_time:.0f} blocks/s")
    else:
        print(f"Grayscale compression: {process_time:.3f}s | PSNR: {psnr:.2f} dB | {total_blocks/process_time:.0f} blocks/s")
    
    return compressed_img, info

def main():
    """
    Main function to test color-preserving DCT compression
    """
    print("DCT Image Compression - RGB and Grayscale")
    print("=" * 50)
    
    # Load or create image
    image = load_or_create_image_color()
    print(f"Image shape: {image.shape}")
    
    if len(image.shape) == 3:
        print(f"Color image with {image.shape[2]} channels")
        print(f"Image size: {image.shape[0] * image.shape[1] * image.shape[2]:,} total values")
    else:
        print("Grayscale image")
        print(f"Image size: {image.shape[0] * image.shape[1]:,} pixels")
    
    # Get compression ratio
    try:
        compression_ratio = float(input("Enter compression ratio (0.01 to 1.0, e.g., 0.1 for 10%): "))
        compression_ratio = max(0.01, min(1.0, compression_ratio))
    except ValueError:
        print("Invalid input. Using default compression ratio of 0.1")
        compression_ratio = 0.1
    
    print(f"\nApplying DCT compression with ratio: {compression_ratio:.3f}")
    
    # Apply compression with benchmarking
    compressed_img, info = benchmark_compression(image, compression_ratio)
    
    # Calculate quality metrics
    psnr = calculate_psnr_color(image, compressed_img)
    
    # Display detailed results
    print(f"\nDetailed Compression Results:")
    print(f"Processing time: {info['processing_time']:.3f} seconds")
    print(f"Original coefficients: {info['original_size']:,}")
    print(f"Kept coefficients: {info['compressed_size']:,}")
    print(f"Space saved: {info['space_saved']:.1%}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Total 8x8 blocks: {info['total_blocks']:,}")
    print(f"Processing speed: {info['total_blocks']/info['processing_time']:.0f} blocks/second")
    
    # Save images to files with compression analysis
    print(f"\nSaving images...")
    saved_paths = save_images_color(image, compressed_img, compression_ratio)
    print(f"Images saved to:")
    print(f"  Original: {saved_paths['original_path']}")
    print(f"  Compressed: {saved_paths['compressed_path']}")
    print(f"  Difference: {saved_paths['difference_path']}")
    
    # Explain the file size results
    print(f"\nWhy might the compressed PNG be larger?")
    print(f"• DCT compression reduces data by {info['space_saved']:.1%}")
    print(f"• But we save the full reconstructed image as PNG")
    print(f"• PNG compression works differently on reconstructed vs original images")
    print(f"• Actual DCT compression is {info['compressed_size']:,} vs {info['original_size']:,} coefficients")
    
    # Show actual compression ratio
    theoretical_compressed_size = info['compressed_size'] * 4  # 4 bytes per float coefficient
    if len(image.shape) == 3:
        original_raw_size = image.shape[0] * image.shape[1] * image.shape[2]
    else:
        original_raw_size = image.shape[0] * image.shape[1]
    
    print(f"\nActual DCT Compression Analysis:")
    print(f"Original raw data: {original_raw_size:,} bytes")
    print(f"DCT compressed data: {theoretical_compressed_size:,} bytes")
    print(f"True compression ratio: {theoretical_compressed_size/original_raw_size:.3f}")
    print(f"Space saved by DCT: {(1 - theoretical_compressed_size/original_raw_size):.1%}")
    
    # Display images (subsample if too large)
    display_image = image
    display_compressed = compressed_img
    
    if image.shape[0] > 800 or image.shape[1] > 800:
        step = max(image.shape[0] // 400, image.shape[1] // 400, 1)
        display_image = image[::step, ::step]
        display_compressed = compressed_img[::step, ::step]
        print(f"\nDisplaying subsampled images (every {step} pixels) for visualization")
    
    # Create appropriate figure layout
    if len(image.shape) == 3:
        # Color image
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top row: Full images
        axes[0, 0].imshow(display_image)
        axes[0, 0].set_title(f'Original Color Image\n{image.shape[0]}x{image.shape[1]} pixels')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(display_compressed)
        axes[0, 1].set_title(f'Compressed\nRatio: {compression_ratio:.3f}, PSNR: {psnr:.1f} dB')
        axes[0, 1].axis('off')
        
        # Difference image (magnitude)
        diff = np.abs(display_image.astype(np.float32) - display_compressed.astype(np.float32))
        diff_magnitude = np.sqrt(np.sum(diff**2, axis=2))
        im = axes[0, 2].imshow(diff_magnitude, cmap='hot', vmin=0, vmax=np.max(diff_magnitude))
        axes[0, 2].set_title('Compression Error (Magnitude)')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], shrink=0.6)
        
        # Bottom row: Individual channel differences
        channel_names = ['Red', 'Green', 'Blue']
        for ch in range(3):
            ch_diff = diff[:, :, ch]
            im = axes[1, ch].imshow(ch_diff, cmap='hot', vmin=0, vmax=np.max(ch_diff))
            axes[1, ch].set_title(f'{channel_names[ch]} Channel Error')
            axes[1, ch].axis('off')
            plt.colorbar(im, ax=axes[1, ch], shrink=0.6)
    else:
        # Grayscale image
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(display_image, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f'Original Image\n{image.shape[0]}x{image.shape[1]} pixels')
        axes[0].axis('off')
        
        axes[1].imshow(display_compressed, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Compressed\nRatio: {compression_ratio:.3f}, PSNR: {psnr:.1f} dB')
        axes[1].axis('off')
        
        # Difference image
        diff = np.abs(display_image.astype(np.float32) - display_compressed.astype(np.float32))
        im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=np.max(diff))
        axes[2].set_title('Compression Error')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], shrink=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Test multiple compression ratios
    print(f"\nTesting different compression ratios:")
    if len(image.shape) == 3:
        print("Ratio | Coeffs/Block (RGB) | Space Saved | PSNR (dB) | Time (s)")
    else:
        print("Ratio | Coeffs/Block | Space Saved | PSNR (dB) | Time (s)")
    print("-" * 75)
    
    ratios = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    for ratio in ratios:
        start_time = time.time()
        comp_img, comp_info = compress_image_dct_color(image, ratio)
        process_time = time.time() - start_time
        psnr_val = calculate_psnr_color(image, comp_img)
        
        coeffs_per_block = comp_info['compressed_size'] // comp_info['total_blocks']
        if len(image.shape) == 3:
            print(f"{ratio:5.2f} | {coeffs_per_block:17d} | {comp_info['space_saved']:10.1%} | {psnr_val:8.2f} | {process_time:7.3f}")
        else:
            print(f"{ratio:5.2f} | {coeffs_per_block:11d} | {comp_info['space_saved']:10.1%} | {psnr_val:8.2f} | {process_time:7.3f}")
    
    # Ask if user wants to save multiple compression levels
    save_multiple = input("\nSave images at multiple compression levels? (y/n): ").strip().lower()
    if save_multiple == 'y':
        print("Saving multiple compression levels...")
        levels = [0.05, 0.1, 0.2, 0.5] if len(image.shape) == 3 else [0.02, 0.05, 0.1, 0.2, 0.5]
        
        for ratio in levels:
            comp_img, _ = compress_image_dct_color(image, ratio)
            save_images_color(image, comp_img, ratio, f"dct_output/ratio_{ratio:.3f}")
        print("Multiple compression levels saved!")

if __name__ == "__main__":
    main()