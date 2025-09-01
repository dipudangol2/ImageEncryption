import sys
import os
import time
import numpy as np
from PIL import Image
from aes_cipher import AESCipher   # <- import your AES class file


def save_image_from_array(array, path):
    """Save numpy array as image file."""
    img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    img.save(path)


def main(image_path, key_string):
    if len(key_string) != 16:
        print("Error: Key must be exactly 16 characters long for AES-128")
        sys.exit(1)
    key_bytes = key_string.encode("utf-8")

    # Load image
    img = Image.open(image_path).convert("RGB")   # ensure RGB
    img_np = np.array(img)
    print(f"Loaded image: {img.size}, mode={img.mode}, shape={img_np.shape}")

    # Flatten into bytes
    flat_bytes = img_np.tobytes()

    # AES cipher
    cipher = AESCipher(key_bytes)

    # Encrypt
    print("Encrypting...")
    t0 = time.time()
    encrypted_data = cipher.encrypt(flat_bytes)
    print(f"Encryption took {time.time() - t0:.2f} sec")

    # Save encrypted visualization as image
    enc_len = len(encrypted_data)
    padded = np.frombuffer(encrypted_data, dtype=np.uint8)

    # Arrange into RGB square image for visualization
    pad_len = (3 - (enc_len % 3)) % 3
    if pad_len:
        padded = np.pad(padded, (0, pad_len), "constant")

    rgb_pixels = padded.reshape(-1, 3)
    side = int(np.ceil(np.sqrt(len(rgb_pixels))))
    pad_pixels = side * side - len(rgb_pixels)
    rgb_pixels = np.pad(rgb_pixels, ((0, pad_pixels), (0, 0)), "constant")
    rgb_img = rgb_pixels.reshape(side, side, 3)

    save_image_from_array(rgb_img, "encrypted_image.png")
    print("Encrypted image saved as encrypted_image.png")

    # Decrypt
    print("Decrypting...")
    t1 = time.time()
    decrypted_bytes = cipher.decrypt(encrypted_data)
    print(f"Decryption took {time.time() - t1:.2f} sec")

    # Convert back to numpy image
    decrypted_np = np.frombuffer(decrypted_bytes, dtype=np.uint8)
    decrypted_np = decrypted_np[: img_np.size]  # remove padding
    decrypted_np = decrypted_np.reshape(img_np.shape)

    save_image_from_array(decrypted_np, "decrypted_image.png")
    print("Decrypted image saved as decrypted_image.png")

    print("\nDone ✅")
    print("Files created:")
    print(" - encrypted_image.png (visualization)")
    print(" - decrypted_image.png (final reconstruction)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <image_path> <16_char_key>")
        sys.exit(1)

    image_path = sys.argv[1]
    key = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found")
        sys.exit(1)

    main(image_path, key)
