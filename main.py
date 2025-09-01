import numpy as np
from PIL import Image
from image_compression import compress_color_image, decompress_color_image
from image_encryption import AESCipher
import sys
import os

def get_standard_quant_matrix():
    # JPEG standard luminance quantization matrix
    return np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)

def save_image_from_array(array, path):
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(path)

def main(image_path, key_string):
    if len(key_string) != 16:
        print("Error: Encryption key must be exactly 16 characters long for AES-128")
        sys.exit(1)
    key_bytes = key_string.encode('utf-8')

    # Step 1: Load image and detect/convert to supported format
    #  # Step 1: Load image as RGB
    # img = Image.open(image_path).convert('RGB')
    # img_np = np.array(img)
    img = Image.open(image_path)
    original_mode = img.mode
    print("Original mode:",original_mode)
    
    # Convert to supported format
    if original_mode == 'L':
        print("Detected grayscale image.")
        img_np = np.array(img)
        num_channels = 1
    elif original_mode == 'RGB':
        print("Detected RGB image.")
        img_np = np.array(img)
        num_channels = 3
    else:
        # Convert unsupported formats
        if original_mode in ['RGBA', 'P', 'CMYK', 'YCbCr', 'LAB', 'HSV']:
            print(f"Converting {original_mode} to RGB.")
            img = img.convert('RGB')
            img_np = np.array(img)
            num_channels = 3
        else:
            print(f"Converting {original_mode} to grayscale.")
            img = img.convert('L')
            img_np = np.array(img)
            num_channels = 1
    
    quant_matrix = get_standard_quant_matrix()

    # Step 2: Compress (manual DCT, per channel)
    print("Compressing image...")
    compressed_data, img_dims, num_channels = compress_color_image(img_np, quant_matrix)

    # Step 3: Save compressed image (DCT only, before encryption)
    compressed_preview = decompress_color_image(compressed_data, img_dims, quant_matrix, num_channels)
    print("Saving compressed image...")
    save_image_from_array(compressed_preview, 'compressed_output.png')

    # Step 4: Encrypt compressed data using manual AES-128 ECB
    print("Encrypting data...")
    cipher = AESCipher(key_bytes)
    compressed_bytes = compressed_data.astype(np.int16).tobytes()
    encrypted_data = cipher.encrypt(compressed_bytes)

    # Step 5: Save encrypted data as binary
    print("Saving encrypted data...")
    with open('encrypted_data.bin', 'wb') as f:
        f.write(encrypted_data)

    # Step 6: Save encrypted data as an image (visualization)
    """enc_len = len(encrypted_data)
    side = int(np.ceil(np.sqrt(enc_len)))
    padded = np.frombuffer(encrypted_data, dtype=np.uint8)
    padded = np.pad(padded, (0, side*side - enc_len), 'constant', constant_values=0)
    enc_img = padded.reshape((side, side))
    save_image_from_array(enc_img, 'encrypted_image.png')"""
    enc_len = len(encrypted_data)
    pad_len = (3 - (enc_len % 3)) % 3
    padded = np.frombuffer(encrypted_data, dtype=np.uint8)
    padded = np.pad(padded, (0, pad_len), 'constant', constant_values=0)
    rgb_pixels = padded.reshape(-1, 3)
    side = int(np.ceil(np.sqrt(len(rgb_pixels))))
    pad_pixels = side * side - len(rgb_pixels)
    rgb_pixels = np.pad(rgb_pixels, ((0, pad_pixels), (0, 0)), 'constant', constant_values=0)
    rgb_img = rgb_pixels.reshape(side, side, 3)
    save_image_from_array(rgb_img, 'encrypted_image.png')

    # Step 7: Decrypt
    print("Decrypting data...")
    with open('encrypted_data.bin', 'rb') as f:
        encrypted_data_read = f.read()
    decrypted_bytes = cipher.decrypt(encrypted_data_read)

    # Step 8: Decompress
    print("Decompressing data...")
    decompressed_coeffs = np.frombuffer(decrypted_bytes, dtype=np.int16)
    decompressed_coeffs = decompressed_coeffs[:compressed_data.size]  # Remove any padding
    reconstructed_img = decompress_color_image(decompressed_coeffs, img_dims, quant_matrix, num_channels)

    # Step 9: Save reconstructed image
    print("Saving reconstructed image...")
    save_image_from_array(reconstructed_img, 'decrypted_image.png')

    print("Process complete.\n"
          "Files saved:\n"
          "  - compressed_output.png (DCT only)\n"
          "  - encrypted_data.bin (binary)\n"
          "  - encrypted_image.png (visualization)\n"
          "  - decrypted_image.png (final output)\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Compression + AES-128 Encryption Demo")
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('key', help='16-character encryption key for AES-128')
    args = parser.parse_args()
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        sys.exit(1)
    main(args.image_path, args.key)