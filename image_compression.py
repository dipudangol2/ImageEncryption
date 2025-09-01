import numpy as np
import math

def dct_compress(image_data, quantization_matrix):
    """
    Compress image data using Discrete Cosine Transform (DCT).
    
    Args:
        image_data: NumPy array representing the image
        quantization_matrix: 8x8 NumPy array for quantizing DCT coefficients
    
    Returns:
        tuple: (compressed_data, image_dimensions)
    """
    # Convert to grayscale if the image is colored
    if len(image_data.shape) == 3:
        # Convert RGB to grayscale using standard weights
        image_data = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])
    
    height, width = image_data.shape
    
    # Pad image to make dimensions divisible by 8
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    
    padded_image = np.zeros((padded_height, padded_width))
    padded_image[:height, :width] = image_data
    
    compressed_blocks = []
    
    # Process image in 8x8 blocks
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            # Extract 8x8 block
            block = padded_image[i:i+8, j:j+8]
            
            # Shift pixel values to center around 0 (subtract 128 for 8-bit images)
            block = block - 128
            
            # Apply 2D DCT
            dct_block = dct_2d(block)
            
            # Quantize the DCT coefficients
            quantized_block = np.round(dct_block / quantization_matrix).astype(int)
            
            # Apply zig-zag scan to serialize the block
            zigzag_data = zigzag_scan(quantized_block)
            compressed_blocks.extend(zigzag_data)
    
    return np.array(compressed_blocks), (height, width)

def dct_decompress(compressed_data, image_dimensions, quantization_matrix):
    """
    Decompress image data using inverse DCT.
    
    Args:
        compressed_data: Serialized, quantized DCT data
        image_dimensions: Original image dimensions (height, width)
        quantization_matrix: 8x8 quantization matrix used for compression
    
    Returns:
        NumPy array: Decompressed image
    """
    height, width = image_dimensions
    
    # Calculate padded dimensions
    padded_height = ((height + 7) // 8) * 8
    padded_width = ((width + 7) // 8) * 8
    
    # Reconstruct image from compressed blocks
    reconstructed_image = np.zeros((padded_height, padded_width))
    
    block_idx = 0
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            # Extract serialized block data (64 coefficients per block)
            block_data = compressed_data[block_idx:block_idx + 64]
            block_idx += 64
            
            # Inverse zig-zag scan to reconstruct 8x8 block
            quantized_block = inverse_zigzag_scan(block_data)
            
            # De-quantize coefficients
            dct_block = quantized_block * quantization_matrix
            
            # Apply inverse 2D DCT
            reconstructed_block = idct_2d(dct_block)
            
            # Shift pixel values back (add 128)
            reconstructed_block = reconstructed_block + 128
            
            # Clamp values to valid pixel range
            reconstructed_block = np.clip(reconstructed_block, 0, 255)
            
            # Place block in the reconstructed image
            reconstructed_image[i:i+8, j:j+8] = reconstructed_block
    
    # Return only the original image portion (remove padding)
    return reconstructed_image[:height, :width]

def compress_color_image(image_data, quantization_matrix):
    """
    Compress a color image (RGB) using DCT for each channel.
    Args:
        image_data: NumPy array (H, W, 3)
        quantization_matrix: 8x8 NumPy array
    Returns:
        compressed_data: 1D np.array of all channels concatenated
        image_dimensions: (height, width)
        num_channels: 3
    """
    if len(image_data.shape) == 2:
        # Grayscale fallback
        compressed, dims = dct_compress(image_data, quantization_matrix)
        return compressed, dims, 1
    height, width, channels = image_data.shape
    compressed_channels = []
    for ch in range(channels):
        compressed, _ = dct_compress(image_data[:,:,ch], quantization_matrix)
        compressed_channels.append(compressed)
    compressed_data = np.concatenate(compressed_channels)
    return compressed_data, (height, width), channels

def decompress_color_image(compressed_data, image_dimensions, quantization_matrix, num_channels):
    """
    Decompress a color image (RGB) from DCT coefficients for each channel.
    Args:
        compressed_data: 1D np.array
        image_dimensions: (height, width)
        quantization_matrix: 8x8 np.array
        num_channels: 3
    Returns:
        image: np.array (H, W, 3)
    """
    height, width = image_dimensions
    total_pixels = height * width
    channel_size = (int(np.ceil(height/8))*8) * (int(np.ceil(width/8))*8)  # padded size
    channel_coeffs = 64 * (channel_size // 64)
    img_channels = []
    for ch in range(num_channels):
        start = ch * channel_coeffs
        end = (ch+1) * channel_coeffs
        channel_coeff = compressed_data[start:end]
        channel_img = dct_decompress(channel_coeff, image_dimensions, quantization_matrix)
        img_channels.append(channel_img)
    img = np.stack(img_channels, axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)

def dct_2d(block):
    """
    Apply 2D Discrete Cosine Transform to an 8x8 block.
    
    Args:
        block: 8x8 NumPy array
    
    Returns:
        8x8 NumPy array: DCT coefficients
    """
    N = 8
    dct_matrix = np.zeros((N, N))
    
    # Create DCT transformation matrix
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_matrix[i, j] = 1 / math.sqrt(N)
            else:
                dct_matrix[i, j] = math.sqrt(2/N) * math.cos((2*j + 1) * i * math.pi / (2*N))
    
    # Apply 2D DCT: DCT_matrix * block * DCT_matrix^T
    return np.dot(np.dot(dct_matrix, block), dct_matrix.T)

def idct_2d(dct_block):
    """
    Apply 2D Inverse Discrete Cosine Transform to an 8x8 block of DCT coefficients.
    
    Args:
        dct_block: 8x8 NumPy array of DCT coefficients
    
    Returns:
        8x8 NumPy array: Reconstructed pixel block
    """
    N = 8
    dct_matrix = np.zeros((N, N))
    
    # Create DCT transformation matrix (same as forward DCT)
    for i in range(N):
        for j in range(N):
            if i == 0:
                dct_matrix[i, j] = 1 / math.sqrt(N)
            else:
                dct_matrix[i, j] = math.sqrt(2/N) * math.cos((2*j + 1) * i * math.pi / (2*N))
    
    # Apply inverse 2D DCT: DCT_matrix^T * dct_block * DCT_matrix
    return np.dot(np.dot(dct_matrix.T, dct_block), dct_matrix)

def zigzag_scan(block):
    """
    Apply zig-zag scan to serialize an 8x8 block into a 1D array.
    This exploits the energy compaction property of DCT.
    
    Args:
        block: 8x8 NumPy array
    
    Returns:
        1D array: Serialized coefficients in zig-zag order
    """
    # Define zig-zag pattern for 8x8 block
    zigzag_order = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    
    return [block[i, j] for i, j in zigzag_order]

def inverse_zigzag_scan(zigzag_data):
    """
    Reconstruct 8x8 block from zig-zag serialized data.
    
    Args:
        zigzag_data: 1D array of 64 coefficients in zig-zag order
    
    Returns:
        8x8 NumPy array: Reconstructed block
    """
    # Define zig-zag pattern for 8x8 block
    zigzag_order = [
        (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
        (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
        (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
        (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
        (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
        (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
        (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
        (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
    ]
    
    block = np.zeros((8, 8))
    for idx, (i, j) in enumerate(zigzag_order):
        block[i, j] = zigzag_data[idx]
    
    return block