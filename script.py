import cv2

# Load your images
img1 = cv2.imread('testimage.jpg')
img2 = cv2.imread('compressed_output.png')

# Print the shapes
print(f"Shape of image 1: {img1.shape}")
print(f"Shape of image 2: {img2.shape}")

print(f"Dtype of image 1: {img1.dtype}")
print(f"Dtype of image 2: {img2.dtype}")

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

# Assuming img1 and img2 are your loaded uint8 images
# Convert to grayscale
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print(f"Shape of grayscale images: {img1_gray.shape}, {img2_gray.shape}") # Should be (1080, 1920)

psnr_value = peak_signal_noise_ratio(img1_gray, img2_gray, data_range=255)
print(f"PSNR (grayscale): {psnr_value}")


from skimage.metrics import peak_signal_noise_ratio as psnr_ski

# Assuming img1 and img2 are loaded and are uint8
# The scikit-image function handles multi-channel images automatically.
try:
    psnr_value = psnr_ski(img1, img2, data_range=255)
    print(f"PSNR value: {psnr_value}")
except Exception as e:
    print(f"An error occurred: {e}")