from api.manual_dct_compressor import ManualDCTCompressor
from api.performance_analyzer import PerformanceAnalyzer
from api.main import save_image_from_array
from PIL import Image
import numpy as np 



compressor = ManualDCTCompressor()
performance_analyzer = PerformanceAnalyzer()
img = Image.open("sample_image.jpg")
img_np =  np.array(img)
compressed_data = compressor.compress(
                img_np, quality=60, use_color=True
            )

reconstructed_image = compressor.decompress(compressed_data)

save_image_from_array(reconstructed_image, "compressed.jpg")


psnr_value = performance_analyzer.calculate_psnr(img_np, reconstructed_image)
print(f"PSNR between original and compressed image: {psnr_value:.2f} dB")