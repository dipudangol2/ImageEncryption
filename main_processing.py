#!/usr/bin/env python3
"""
Main processing script for image compression and encryption.
This script orchestrates the entire workflow of loading, compressing, encrypting,
decrypting, and decompressing images.
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
from image_compression import compress_color_image, decompress_color_image
from image_encryption import AESCipher

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
    img = Image.fromarray(np.uint8(array))
    img.save(path)

def main(image_path, key_string):
    if len(key_string) != 16:
        print("Error: Encryption key must be exactly 16 characters long for AES-128")
        sys.exit(1)
    key_bytes = key_string.encode('utf-8')
    # Step 1: Load image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    quant_matrix = get_standard_quant_matrix()
    # Step 2: Compress (color DCT)
    compressed_data, img_dims, num_channels = compress_color_image(img_np, quant_matrix)
    # Step 3: Encrypt
    cipher = AESCipher(key_bytes)
    # Use int16 for DCT coefficients
    compressed_bytes = compressed_data.astype(np.int16).tobytes()
    encrypted_data = cipher.encrypt(compressed_bytes)
    # Step 4: Save encrypted data as binary
    with open('encrypted_data.bin', 'wb') as f:
        f.write(encrypted_data)
    # Step 5: Save encrypted data as an image (visualization)
    enc_len = len(encrypted_data)
    side = int(np.ceil(np.sqrt(enc_len)))
    padded = np.frombuffer(encrypted_data, dtype=np.uint8)
    padded = np.pad(padded, (0, side*side - enc_len), 'constant', constant_values=0)
    enc_img = padded.reshape((side, side))
    save_image_from_array(enc_img, 'encrypted_image.png')
    # Step 6: Decrypt
    with open('encrypted_data.bin', 'rb') as f:
        encrypted_data_read = f.read()
    decrypted_bytes = cipher.decrypt(encrypted_data_read)
    # Step 7: Decompress
    decompressed_coeffs = np.frombuffer(decrypted_bytes, dtype=np.int16)
    decompressed_coeffs = decompressed_coeffs[:compressed_data.size]  # Remove any padding
    reconstructed_img = decompress_color_image(decompressed_coeffs, img_dims, quant_matrix, num_channels)
    # Step 8: Save reconstructed image
    save_image_from_array(reconstructed_img, 'decrypted_image.png')
    print("Process complete. Files saved: encrypted_data.bin, encrypted_image.png, decrypted_image.png")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Image Compression and Encryption Tool (Manual DCT + AES-128)")
    parser.add_argument('image_path', help='Path to the input image file')
    parser.add_argument('key', help='16-character encryption key for AES-128')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist")
        sys.exit(1)
    print("Image Compression and Encryption Tool (Manual DCT + AES-128)")
    print("=====================================")
    main(args.image_path, args.key)
