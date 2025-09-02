# DCT2DFF.py Integration Plan

## Overview
Replace the current API compression system (improved_compressor.py + unified_compression.py) with DCT2DFF.py to achieve:
- ✅ Manual DCT implementation (no scipy dependency)
- ✅ No quantization (ratio-based coefficient selection)
- ✅ No zigzag indices (direct coefficient processing)

## Implementation Strategy

### Phase 1: Testing DCT2DFF.py
Create `test_dct2dff.py` to verify:
1. **Basic functionality**: Load sample image → compress → decompress → verify quality
2. **Both color and grayscale**: Test with RGB and grayscale images
3. **Different compression ratios**: Test 0.1, 0.2, 0.5 ratios
4. **Performance benchmarking**: Measure speed and PSNR
5. **API compatibility**: Ensure output format works with existing serialization

### Phase 2: Create DCT2DFF API Wrapper
Create `api/dct2dff_compressor.py` with:
```python
class DCT2DFFCompressor:
    def compress(self, image: np.ndarray, quality: int = 75, use_color: bool = True) -> Dict
    def decompress(self, compressed_data: Dict) -> np.ndarray
    def get_compression_stats(self, original_image: np.ndarray, compressed_data: Dict) -> Dict
```

**Key Design Decisions:**
- **Quality to Ratio Conversion**: Map quality (1-100) to compression_ratio (0.01-1.0)
  - `compression_ratio = quality / 100.0` (simple linear mapping)
- **Data Format**: Store coefficients per block with position metadata for reconstruction
- **Backward Compatibility**: Maintain same API interface as UnifiedCompressor

### Phase 3: Integration with API
Update `api/main.py`:
1. Replace `from unified_compression import UnifiedCompressor`
2. Import new `from dct2dff_compressor import DCT2DFFCompressor`
3. Update compressor instantiation: `compressor = DCT2DFFCompressor()`
4. Verify all endpoints work correctly

## Technical Design

### DCT2DFF Functions to Use:
- `compress_image_dct_color()` - Main compression function
- `calculate_psnr_color()` - Quality metrics
- Core DCT functions are already optimized

### Data Format Strategy:
```python
# For API compatibility, we need to store compressed data that can be:
# 1. Serialized with pickle for encryption
# 2. Reconstructed without losing quality information
# 3. Include metadata for proper decompression

compressed_data = {
    'dct_coefficients': [  # List of block data
        {
            'position': (i, j),      # Block position
            'indices': indices,      # Coefficient positions
            'coefficients': coeffs   # Actual values
        }
    ],
    'original_shape': (h, w) or (h, w, 3),
    'compression_ratio': ratio,
    'num_channels': 1 or 3,
    'use_manual_dct': True,  # Flag for identification
    # Metadata for API compatibility
    'original_format': '.png',
    'original_filename': 'image.png'
}
```

### Quality Mapping Strategy:
```python
def quality_to_compression_ratio(quality: int) -> float:
    """
    Convert API quality (1-100) to DCT2DFF compression ratio (0.01-1.0)
    
    Quality 100 = ratio 1.0 (no compression, all coefficients)
    Quality 50  = ratio 0.5 (50% of coefficients)
    Quality 10  = ratio 0.1 (10% of coefficients)
    Quality 1   = ratio 0.01 (1% of coefficients)
    """
    return max(0.01, min(1.0, quality / 100.0))
```

### Compression Function Adaptation:
```python
def compress(self, image: np.ndarray, quality: int = 75, use_color: bool = True) -> Dict:
    # Convert quality to compression ratio
    compression_ratio = quality / 100.0
    
    # Use DCT2DFF compression
    if len(image.shape) == 2:  # Grayscale
        compressed_img, coeffs_list, info = compress_channel_dct_with_storage(
            image, compression_ratio
        )
        return {
            'dct_coefficients': coeffs_list,
            'original_shape': image.shape,
            'compression_ratio': compression_ratio,
            'num_channels': 1,
            'use_manual_dct': True
        }
    else:  # Color
        # Process each channel separately and collect coefficients
        all_coefficients = []
        for channel_idx in range(3):
            channel = image[:, :, channel_idx]
            _, coeffs_list, _ = compress_channel_dct_with_storage(
                channel, compression_ratio
            )
            all_coefficients.append(coeffs_list)
        
        return {
            'dct_coefficients': all_coefficients,
            'original_shape': image.shape,
            'compression_ratio': compression_ratio,
            'num_channels': 3,
            'use_manual_dct': True
        }
```

### Decompression Function Adaptation:
```python
def decompress(self, compressed_data: Dict) -> np.ndarray:
    # Reconstruct from stored DCT coefficients
    original_shape = compressed_data['original_shape']
    coeffs = compressed_data['dct_coefficients']
    ratio = compressed_data['compression_ratio']
    
    if compressed_data['num_channels'] == 1:
        # Grayscale reconstruction
        return self._reconstruct_from_coefficients(coeffs, original_shape, ratio)
    else:
        # Color reconstruction - process each channel
        channels = []
        for channel_coeffs in coeffs:
            channel = self._reconstruct_from_coefficients(
                channel_coeffs, original_shape[:2], ratio
            )
            channels.append(channel)
        return np.stack(channels, axis=2)
```

## Benefits of This Approach

1. **✅ Manual DCT**: Uses pre-computed matrices, no scipy.fft dependency
2. **✅ No Quantization**: Pure coefficient magnitude-based compression
3. **✅ No Zigzag**: Direct 8x8 block processing
4. **✅ API Compatible**: Same interface, drop-in replacement
5. **✅ Maintains Quality**: Same PSNR calculation and metrics
6. **✅ Efficient**: Pre-computed DCT matrices for better performance

## Migration Steps

1. **Test DCT2DFF.py**: Verify functionality with sample images
2. **Create Wrapper**: Build `api/dct2dff_compressor.py` 
3. **Test Wrapper**: Ensure API compatibility
4. **Update main.py**: Replace compression import
5. **Integration Test**: Full API test with encryption/decryption
6. **Performance Verification**: Compare speed and quality metrics

## File Structure After Integration
```
api/
├── dct2dff_compressor.py    # New: DCT2DFF API wrapper
├── main.py                  # Updated: Use new compressor
├── aes_cipher.py           # Unchanged
├── file_manager.py         # Unchanged
├── improved_compressor.py  # Can be removed after verification
└── unified_compression.py  # Can be removed after verification
```

## Testing Strategy

### Unit Tests:
- Test compression ratio conversion (quality → ratio)
- Test data format compatibility with pickle serialization
- Test reconstruction accuracy (PSNR validation)

### Integration Tests:
- Full API encrypt/decrypt workflow
- File size analysis
- Performance benchmarking vs current implementation

### Acceptance Criteria:
- ✅ Manual DCT implementation (no scipy)
- ✅ No quantization matrices
- ✅ No zigzag indices
- ✅ Same API interface compatibility
- ✅ Maintained or improved compression quality
- ✅ Proper encryption/decryption workflow

## Risk Mitigation

1. **Backup Current Implementation**: Keep existing files until verification complete
2. **Gradual Migration**: Test each component independently
3. **Performance Monitoring**: Ensure no degradation in speed/quality
4. **API Compatibility**: Verify all endpoints work with new compression

This plan ensures a smooth transition from the current JPEG-style compression to the pure DCT implementation requested.