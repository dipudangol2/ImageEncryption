import time
import numpy as np
from PIL import Image
from aes_cipher import AESCipher
import sys
import os
import pickle
from optimized_dct_compression import OptimizedDCTCompressor

from improved_compressor import ImprovedDCTCompressor

# from manual_dct_compression import ManualDCTCompressor as ImprovedDCTCompressor


def save_image_from_array(array, path):
    """Save numpy array as image file."""
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(path)


def serialize_compressed_data(compressed_data):
    """Serialize compressed data dictionary to bytes for encryption."""
    return pickle.dumps(compressed_data)


def deserialize_compressed_data(data_bytes):
    """Deserialize bytes back to compressed data dictionary."""
    return pickle.loads(data_bytes)


def main(image_path, key_string, quality=75):
    if len(key_string) != 16:
        print("Error: Encryption key must be exactly 16 characters long for AES-128")
        sys.exit(1)
    key_bytes = key_string.encode("utf-8")

    # Initialize the optimized DCT compressor
    compressor = ImprovedDCTCompressor()

    # Step 1: Load image and detect/convert to supported format
    img = Image.open(image_path)
    original_mode = img.mode
    print(f"Original mode: {original_mode}")
    print(f"Original size: {img.size}")

    # Convert to supported format
    if original_mode == "L":
        print("Detected grayscale image.")
        img_np = np.array(img)
        use_color = False
    elif original_mode == "RGB":
        print("Detected RGB image.")
        img_np = np.array(img)
        use_color = True
    else:
        # Convert unsupported formats
        if original_mode in ["RGBA", "P", "CMYK", "YCbCr", "LAB", "HSV"]:
            print(f"Converting {original_mode} to RGB.")
            img = img.convert("RGB")
            img_np = np.array(img)
            use_color = True
        else:
            print(f"Converting {original_mode} to grayscale.")
            img = img.convert("L")
            img_np = np.array(img)
            use_color = False

    print(f"Image shape: {img_np.shape}")
    print(f"Compression quality: {quality}")
    file_ext = os.path.splitext(image_path)[1].lower()
    print(file_ext)

    # Step 2: Compress using optimized DCT
    print("Compressing image with optimized DCT...")
    compressed_data = compressor.compress(img_np, quality=quality, use_color=use_color)

    # Step 3: Save compressed image preview (before encryption)
    print("Generating compressed preview...")
    compressed_preview = compressor.decompress(compressed_data)
    save_image_from_array(compressed_preview, "compressed_output.png")

    # Calculate compression statistics
    original_size = img_np.nbytes
    compressed_bytes = serialize_compressed_data(compressed_data)
    compressed_size = len(compressed_bytes)
    compression_ratio = original_size / compressed_size

    print(f"Original size: {original_size:,} bytes")
    print(f"Compressed size: {compressed_size:} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Calculate image quality metrics
    mse = np.mean((img_np.astype(float) - compressed_preview.astype(float)) ** 2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse) if mse > 0 else float("inf")
    print(f"MSE: {mse:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    first_time = time.time()

    # Step 4: Encrypt compressed data using AES-128
    print("Encrypting compressed data...")
    cipher = AESCipher(key_bytes)
    encrypted_data = cipher.encrypt(compressed_bytes)
    Aes_time = time.time() - first_time
    print(f"for aes encryption Aes time: {Aes_time:.2f} seconds")
    imagetrack = time.time()

    # Step 5: Save encrypted data as binary
    print("Saving encrypted binary data...")
    with open("encrypted_data.bin", "wb") as f:
        f.write(encrypted_data)

    # Step 6: Save encrypted data as an image visualization
    print("Creating encrypted data visualization...")
    enc_len = len(encrypted_data)

    # Create RGB visualization of encrypted data
    print("Creating encrypted data visualization...")
    enc_len = len(encrypted_data)

    if use_color:  # If the original image was color
        # Create RGB visualization of encrypted data
        pad_len = (3 - (enc_len % 3)) % 3
        padded = np.frombuffer(encrypted_data, dtype=np.uint8)
        padded = np.pad(padded, (0, pad_len), "constant", constant_values=0)
        rgb_pixels = padded.reshape(-1, 3)
        side = int(np.ceil(np.sqrt(len(rgb_pixels))))
        pad_pixels = side * side - len(rgb_pixels)
        rgb_pixels = np.pad(
            rgb_pixels, ((0, pad_pixels), (0, 0)), "constant", constant_values=0
        )
        rgb_img = rgb_pixels.reshape(side, side, 3)
        save_image_from_array(rgb_img, "encrypted_image.png")
    else:  # If the original image was grayscale
        # Create grayscale visualization of encrypted data
        side = int(np.ceil(np.sqrt(enc_len)))
        pad_len = side * side - enc_len
        padded = np.frombuffer(encrypted_data, dtype=np.uint8)
        padded = np.pad(padded, (0, pad_len), "constant", constant_values=0)
        gray_img = padded.reshape(side, side)
        save_image_from_array(gray_img, "encrypted_image.png")

    encrypted_image_time = time.time() - imagetrack
    print(f"for encrypted image time: {encrypted_image_time:.2f} seconds")

    # pad_len = (3 - (enc_len % 3)) % 3
    # padded = np.frombuffer(encrypted_data, dtype=np.uint8)
    # padded = np.pad(padded, (0, pad_len), 'constant', constant_values=0)
    # rgb_pixels = padded.reshape(-1, 3)
    # side = int(np.ceil(np.sqrt(len(rgb_pixels))))
    # pad_pixels = side * side - len(rgb_pixels)
    # rgb_pixels = np.pad(rgb_pixels, ((0, pad_pixels), (0, 0)), 'constant', constant_values=0)
    # rgb_img = rgb_pixels.reshape(side, side, 3)
    # save_image_from_array(rgb_img, 'encrypted_image.png')
    # encrypted_image_time = time.time() - imagetrack
    # print(f"for encrypted image time: {encrypted_image_time:.2f} seconds")

    before_decryption = time.time()
    # Step 7: Decrypt data
    print("Decrypting data...")
    with open("encrypted_data.bin", "rb") as f:
        encrypted_data_read = f.read()
    decrypted_bytes = cipher.decrypt(encrypted_data_read)

    decryption_time = time.time() - before_decryption
    print(f"Decryption time: {decryption_time}sec")
    # Step 8: Deserialize and decompress
    print("Decompressing image...")
    try:
        decrypted_compressed_data = deserialize_compressed_data(decrypted_bytes)
        reconstructed_img = compressor.decompress(decrypted_compressed_data)
    except Exception as e:
        print(f"Error during decompression: {e}")
        print("This might indicate decryption failure or data corruption.")
        return

    # Step 9: Save reconstructed image
    print("Saving final reconstructed image...")
    save_image_from_array(reconstructed_img, "decrypted_image.png")

    # Final quality check
    final_mse = np.mean((img_np.astype(float) - reconstructed_img.astype(float)) ** 2)
    final_psnr = (
        20 * np.log10(255) - 10 * np.log10(final_mse) if final_mse > 0 else float("inf")
    )

    print("\n" + "=" * 50)
    print("PROCESS COMPLETE")
    print("=" * 50)
    print("Files created:")
    print("  - compressed_output.png (DCT compressed preview)")
    print("  - encrypted_data.bin (encrypted binary data)")
    print("  - encrypted_image.png (encrypted data visualization)")
    print("  - decrypted_image.png (final reconstructed image)")
    print()
    print("Statistics:")
    print(f"  - Original size: {original_size:,} bytes")
    print(f"  - Compressed size: {compressed_size:,} bytes")
    print(f"  - Compression ratio: {compression_ratio:.2f}x")
    print(f"  - Final MSE: {final_mse:.2f}")
    print(f"  - Final PSNR: {final_psnr:.2f} dB")

    # Quality assessment
    if final_psnr > 40:
        quality_assessment = "Excellent"
    elif final_psnr > 30:
        quality_assessment = "Good"
    elif final_psnr > 20:
        quality_assessment = "Fair"
    else:
        quality_assessment = "Poor"

    print(f"  - Quality assessment: {quality_assessment}")


def print_usage():
    """Print usage information."""
    print("Usage: python script.py <image_path> <16_char_key> [quality]")
    print()
    print("Arguments:")
    print("  image_path    : Path to input image file")
    print("  16_char_key   : Exactly 16 characters for AES-128 encryption")
    print("  quality       : Optional. Compression quality 1-100 (default: 75)")
    print("                 Higher values = better quality, larger file size")
    print()
    print("Examples:")
    print("  python script.py photo.jpg 'mySecretKey12345'")
    print("  python script.py photo.jpg 'mySecretKey12345' 90")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Image Compression + AES-128 Encryption",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg 'mySecretKey12345'
  %(prog)s photo.jpg 'mySecretKey12345' --quality 90
  
Quality Guide:
  1-30   : High compression, lower quality (good for thumbnails)
  30-70  : Balanced compression and quality (recommended)
  70-100 : Lower compression, high quality (good for archival)
        """,
    )

    parser.add_argument("image_path", help="Path to the input image file")
    parser.add_argument("key", help="16-character encryption key for AES-128")
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=75,
        choices=range(1, 101),
        help="Compression quality 1-100 (default: 75)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        sys.exit(1)

    if len(args.key) != 16:
        print("Error: Encryption key must be exactly 16 characters long for AES-128")
        print(f"Provided key length: {len(args.key)}")
        sys.exit(1)

    try:
        main(args.image_path, args.key, args.quality)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
