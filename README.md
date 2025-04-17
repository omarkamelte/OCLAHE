# O-CLAHE: Official Implementation

[![DOI](https://img.shields.io/badge/DOI-10.21608%2Fijci.2023.212239.1111-blue)](https://doi.org/10.21608/ijci.2023.212239.1111)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of the paper:  
**"An Automated Contrast Enhancement Technique for Remote Sensed Images"**  
Published in the International Journal of Computers and Information (IJCI), 2024.

## Abstract

Remote sensing images often exhibit lower contrast than usual. Contrast Limited Adaptive Histogram Equalization (CLAHE) is a well-established and robust local contrast enhancement algorithm, renowned for its high-quality results, particularly in the medical domain. In this article, we introduce an automated contrast-limit adaptive histogram equalization method applied on remote sensing images, drawing inspiration from CLAHE. 

Our proposed algorithm incorporates automatic outlier detection blocks into the standard CLAHE framework, addressing the limitation of relying on a predetermined single clip-limit as is preset in traditional CLAHE. Instead, our approach adapts multiple clip-limits, one for each Contextual Region (CR), rather than a global clip-limit used in the original algorithm. 

First, the algorithm divides the image into tiles based on proximity, called contextual region, then it computes the histogram for each CR. In this stage, the CLAHE depends on clip-limit as pre-set user input to clip the intensities above clip-limit, then it redistributes the deducted intensities over the quantization range. On the contrary, the proposed algorithm calculates outliers of each CR and considers a clip-limit for the CR. Next, histogram equalization is performed on the modified CR histogram. Finally, Image is reconstructed by applying bilinear interpolation to outcome CRs. 

Experimental and comparison results showed that the proposed technique provides better results than classic CLAHE for remote sensing images on DOTA dataset. Moreover, the proposed algorithm achieves an improvement of 27.960 for PSNR and 3.271 for CG.

## Features

- **GPU Acceleration**: Utilizes PyTorch for efficient GPU processing when available
- **Dynamic Clip Limit**: Automatically detects outliers in each contextual region
- **LAB Color Space Processing**: Enhances only the luminance channel while preserving color information
- **Fully Vectorized Operations**: Maximizes performance through optimized tensor operations
- **Bilinear Interpolation**: Smooth reconstruction of enhanced contextual regions

## Installation

```bash
# Clone this repository
git clone https://github.com/username/pytorch-oclahe.git
cd pytorch-oclahe

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch 1.8.0+
- NumPy
- Matplotlib
- OpenCV (cv2)

## Usage

### Basic Usage with PyTorch

```python
from oclahe import EnhancedPyTorchOCLAHE
import numpy as np

# Initialize the enhanced O-CLAHE processor
clahe = EnhancedPyTorchOCLAHE(target_tile_size=(32, 32), use_gpu=True)

# Load and enhance a grayscale image
img = np.load('remote_sensing_image.npy')
enhanced_img = clahe.apply(img, verbose=True)
```

### Enhancing Color Images (LAB Color Space)

```python
import cv2
from oclahe import EnhancedPyTorchOCLAHE

# Helper functions for LAB color space processing
def getLuminanceLayer(image_bgr):
    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    return l_channel, a_channel, b_channel

def getRGBfromLAB(l, a, b):
    lab_image = cv2.merge((l.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)))
    bgr_image_reconstructed = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)
    return bgr_image_reconstructed

# Load a color image
image = cv2.imread('remote_sensing_color.tif')

# Convert to LAB color space and extract L channel
l_channel, a_channel, b_channel = getLuminanceLayer(image)

# Create EnhancedPyTorchOCLAHE processor
clahe = EnhancedPyTorchOCLAHE(use_gpu=True, target_tile_size=(32, 32))

# Apply the enhanced CLAHE to the Luminance layer only
enhanced_l = clahe.apply(l_channel, verbose=True)

# Merge the enhanced Luminance layer back with the a and b channels
enhanced_image = getRGBfromLAB(enhanced_l, a_channel, b_channel)

# Save the enhanced image
cv2.imwrite('enhanced_remote_sensing.png', enhanced_image)
```

### Batch Processing

```python
import os
import cv2
import numpy as np
from oclahe import EnhancedPyTorchOCLAHE

def process_directory(input_dir, output_dir, tile_size=(32, 32), use_gpu=True):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor once for efficiency
    clahe = EnhancedPyTorchOCLAHE(target_tile_size=tile_size, use_gpu=use_gpu)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            # Load image
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            
            # Process image
            l, a, b = getLuminanceLayer(image)
            enhanced_l = clahe.apply(l)
            enhanced_image = getRGBfromLAB(enhanced_l, a, b)
            
            # Save result
            output_path = os.path.join(output_dir, f"enhanced_{filename}")
            cv2.imwrite(output_path, enhanced_image)
            
    print(f"Processed {len(os.listdir(input_dir))} images")

# Example usage
process_directory("input_images", "enhanced_images")
```

## Algorithm Overview

![image](https://github.com/user-attachments/assets/e8628dc6-b4b1-401a-9798-ebdadeacfa30)

The EnhancedPyTorchOCLAHE implements an optimized version of the O-CLAHE algorithm with these key steps:

1. **Dynamic Clip Limit Detection**: 
   - Uses Interquartile Range (IQR) method to identify outliers in each contextual region's histogram
   - Calculates a specific clip limit for each region based on the outlier statistics

2. **Grid-Based Processing**:
   - Divides the image into tiles (contextual regions)
   - Processes each region independently with its own calculated clip limit

3. **Histogram Processing**:
   - Creates histogram for each tile
   - Applies dynamic clipping 
   - Redistributes excess counts among non-clipped bins

4. **Interpolation**:
   - Uses bilinear interpolation to eliminate tile boundary artifacts
   - Ensures smooth transitions between adjacent contextual regions

5. **Color Preservation**:
   - For color images, only the luminance (L) channel is enhanced in LAB color space
   - Preserves original color information in the a and b channels

## Performance Optimization

The implementation includes several optimizations to maximize performance:

- **GPU Acceleration**: Utilizes CUDA when available for parallel processing
- **Vectorized Operations**: Uses PyTorch's tensor operations for efficient computation
- **Memory Efficiency**: Minimizes unnecessary tensor allocations
- **Adaptive Processing**: Adjusts parameters based on input image characteristics

## Results

The proposed method outperforms traditional CLAHE on remote sensing images from the DOTA dataset:

| Metric | Traditional CLAHE | Our Method | Improvement |
|--------|-------------------|------------|------------|
| PSNR   | Baseline          | +27.960    | 27.960     |
| CG     | Baseline          | +3.271     | 3.271      |

## Gray remote sensed images visual contrast enhancement performance

![image](https://github.com/user-attachments/assets/d298fc45-ad7b-4a33-bfd0-821ba5fd8170)

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{kamel2024automated,
  author = {Kamel, Omar and Amin, Khalid and Semary, Noura and Aboelenien, Nagwa},
  title = {An Automated Contrast Enhancement Technique for Remote Sensed Images},
  journal = {IJCI. International Journal of Computers and Information},
  volume = {11},
  number = {1},
  pages = {1-16},
  year  = {2024},
  publisher = {Minufiya University; Faculty of Computers and Information},
  issn = {1687-7853},
  eissn = {2735-3257},
  doi = {10.21608/ijci.2023.212239.1111},
  url = {https://ijci.journals.ekb.eg/article_318076.html}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- Omar Kamel
- Khalid Amin
- Noura Semary
- Nagwa Aboelenien

## Acknowledgements

We would like to thank Minufiya University, Faculty of Computers and Information for their support during this research work. Additionally, we acknowledge the creators of the DOTA dataset for providing the benchmark images used in our evaluation.
