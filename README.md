# 🖼️ Computer Vision and Image Processing (CVIP) Mini Project

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5.4+-green.svg)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org)
[![License](https://img.shields.io/badge/License-Educational-yellow.svg)](#)

A comprehensive Python implementation of fundamental computer vision and image processing techniques, featuring manual implementations of color space conversions, various filtering operations, and Fourier domain processing. This project demonstrates core CV algorithms built from scratch using OpenCV and NumPy.

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Features](#-features)
- [🛠️ Technical Implementation](#️-technical-implementation)
- [📊 Results and Analysis](#-results-and-analysis)
- [🚀 Getting Started](#-getting-started)
- [📁 Project Structure](#-project-structure)
- [🔬 Detailed Algorithm Explanations](#-detailed-algorithm-explanations)
- [📈 Performance Analysis](#-performance-analysis)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Project Overview

This project implements three major components of computer vision and image processing:

| **Part** | **Focus Area** | **Techniques Implemented** |
|----------|----------------|----------------------------|
| **Part A** | Color Space Conversions | RGB → HSV (2 methods), RGB → CMYK, RGB → LAB |
| **Part B** | Spatial Domain Filtering | Convolution, Averaging, Gaussian, Median, Brightness/Contrast |
| **Part C** | Frequency Domain Processing | DFT, Fourier Domain Filtering, IDFT |

## ✨ Features

### 🎨 Color Space Conversions (Part A)

#### 1. RGB to HSV Conversion
- **Method 1**: Standard HSV conversion using min-max approach
- **Method 2**: Alternative HSV conversion using class-taught formula with trigonometric functions

```python
# HSV Conversion Formula (Method 1)
V = max(R, G, B)
S = (V - min(R, G, B)) / V  (if V ≠ 0)
H = calculated based on which channel has maximum value
```

#### 2. RGB to CMYK Conversion
Implements the subtractive color model conversion:

```python
# CMYK Conversion Formula
C = 1 - R/255
M = 1 - G/255  
Y = 1 - B/255
K = min(C, M, Y)
```

#### 3. RGB to LAB Conversion
CIE LAB color space conversion with proper illuminant normalization:

```python
# LAB Conversion (simplified)
XYZ = RGB × Conversion_Matrix
L* = 116 × f(Y/Yn) - 16
a* = 500 × [f(X/Xn) - f(Y/Yn)]
b* = 200 × [f(Y/Yn) - f(Z/Zn)]
```

### 🔍 Image Filtering Operations (Part B)

| **Filter Type** | **Kernel Size** | **Purpose** | **Use Case** |
|-----------------|-----------------|-------------|--------------|
| **Convolution** | 3×3 | General filtering | Edge detection, blurring |
| **Average** | 3×3 | Noise reduction | Simple smoothing |
| **Gaussian** | 3×3 | Smooth blurring | Noise reduction with edge preservation |
| **Median** | 5×5 | Salt & pepper noise removal | Non-linear filtering |

#### Filter Kernels Used:

```python
# Average Filter (1/9 normalization)
[1, 1, 1]
[1, 1, 1] × (1/9)
[1, 1, 1]

# Gaussian Filter (1/16 normalization)  
[1, 2, 1]
[2, 4, 2] × (1/16)
[1, 2, 1]
```

### 🌊 Fourier Domain Processing (Part C)

#### 1. Discrete Fourier Transform (DFT)
- Converts spatial domain image to frequency domain
- Visualizes magnitude spectrum with logarithmic scaling
- Applies FFT shift for centered frequency display

#### 2. Gaussian Filtering in Frequency Domain
- Creates custom Gaussian kernel in frequency domain
- Applies low-pass filtering to remove high-frequency noise
- Converts back to spatial domain using Inverse DFT

## 📊 Results and Analysis

### Input Images Used:
- **Lenna.png**: Standard test image for color space conversions
- **Noisy_image.png**: Image with noise for filtering demonstrations  
- **Uexposed.png**: Underexposed image for brightness/contrast adjustment

### Output Analysis:

| **Operation** | **Input** | **Output** | **Key Observation** |
|---------------|-----------|------------|-------------------|
| HSV Conversion (Method 1) | Lenna.png | hsv_image_1.png | Standard HSV representation |
| HSV Conversion (Method 2) | Lenna.png | hsv_image_2.png | Alternative mathematical approach |
| CMYK Conversion | Lenna.png | cmyk_image.png | Subtractive color model |
| LAB Conversion | Lenna.png | lab_image.png | Perceptually uniform color space |
| Convolution Filter | Noisy_image.png | convolved_image.png | Smoothed with edge preservation |
| Average Filter | Noisy_image.png | average_image.png | Simple noise reduction |
| Gaussian Filter | Noisy_image.png | gaussian_image.jpg | Optimal smoothing |
| Median Filter | Noisy_image.png | median_image.jpg | Salt & pepper noise removal |
| Brightness/Contrast | Uexposed.png | adjusted_image.png | Enhanced visibility |
| Fourier Transform | Noisy_image.png | converted_fourier.png | Frequency domain representation |
| Fourier Gaussian | Noisy_image.png | gaussian_fourier.png | Frequency domain filtering |

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy matplotlib
```

### Installation & Usage

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cvip-mini-project
```

2. **Ensure input images are present:**
   - `Lenna.png` (standard test image)
   - `Noisy_image.png` (image with noise)
   - `Uexposed.png` (underexposed image)

3. **Run the main script:**
```bash
python coding.py
```

4. **View results:**
   - Processed images will be saved in the current directory
   - Images will also be displayed using OpenCV windows
   - Press any key to proceed through the image displays

## 📁 Project Structure

```
cvip-mini-project/
│
├── coding.py                 # Main implementation file
├── README.md                # This documentation
├── LICENSE                  # License file
│
├── Input Images/
│   ├── Lenna.png           # Standard test image
│   ├── Noisy_image.png     # Image with noise
│   └── Uexposed.png        # Underexposed image
│
└── results/                # Output directory
    ├── hsv_image_1.png     # HSV conversion (method 1)
    ├── hsv_image_2.png     # HSV conversion (method 2)  
    ├── cmyk_image.png      # CMYK conversion
    ├── lab_image.png       # LAB conversion
    ├── convolved_image.png # Convolution result
    ├── average_image.png   # Average filter result
    ├── gaussian_image.jpg  # Gaussian filter result
    ├── median_image.jpg    # Median filter result
    ├── adjusted_image.png  # Brightness/contrast adjusted
    ├── converted_fourier.png # Fourier domain image
    └── gaussian_fourier.png  # Fourier domain filtered
```

## 🔬 Detailed Algorithm Explanations

### Color Space Conversion Algorithms

#### HSV Conversion (Method 1 - Min-Max Approach)
```python
def rgb_to_hsv_method1(r, g, b):
    # Normalize RGB values
    r, g, b = r/255.0, g/255.0, b/255.0
    
    # Calculate Value (V)
    v = max(r, g, b)
    
    # Calculate Saturation (S)
    if v == 0:
        s = 0
    else:
        s = (v - min(r, g, b)) / v
    
    # Calculate Hue (H)
    if v == r:
        h = 60 * (g - b) / (v - min(r, g, b))
    elif v == g:
        h = 120 + 60 * (b - r) / (v - min(r, g, b))
    elif v == b:
        h = 240 + 60 * (r - g) / (v - min(r, g, b))
    
    return h, s, v
```

#### HSV Conversion (Method 2 - Trigonometric Approach)
```python
def rgb_to_hsv_method2(r, g, b):
    # Alternative approach using trigonometric functions
    v = (r + g + b) / 3
    s = 1 - ((3/(r + g + b)) * min(r, g, b))
    
    theta = arccos((0.5 * ((r-g) + (r-b))) / 
                   sqrt((r-g)² + (r-b)(g-b)))
    
    h = theta if b <= g else 360 - theta
    
    return h, s, v
```

### Filtering Algorithms

#### Convolution Implementation
```python
def convolution(image, kernel):
    # Flip kernel for convolution
    flipped_kernel = np.flip(np.flip(kernel, 0), 1)
    
    # Apply convolution with proper padding
    for i in range(pad_h, height - pad_h):
        for j in range(pad_w, width - pad_w):
            region = image[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1]
            result[i,j] = np.sum(region * flipped_kernel)
    
    return result
```

#### Median Filter Implementation
```python
def median_filter(image, kernel_size):
    for i in range(k_size//2, height - k_size//2):
        for j in range(k_size//2, width - k_size//2):
            neighborhood = image[i-k_size//2:i+k_size//2+1, 
                               j-k_size//2:j+k_size//2+1]
            result[i,j] = np.median(neighborhood)
    
    return result
```

### Fourier Domain Processing

#### DFT and Visualization
```python
def fourier_transform_visualization(image):
    # Convert to frequency domain
    f_transform = cv2.dft(image.astype(np.float32), 
                         flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Shift zero frequency to center
    f_shift = np.fft.fftshift(f_transform)
    
    # Calculate magnitude spectrum
    magnitude = cv2.magnitude(f_shift[:,:,0], f_shift[:,:,1])
    
    # Apply logarithmic scaling for visualization
    log_magnitude = np.log(magnitude + 1)
    
    return log_magnitude
```

## 📈 Performance Analysis

### Computational Complexity

| **Algorithm** | **Time Complexity** | **Space Complexity** | **Notes** |
|---------------|-------------------|-------------------|-----------|
| Color Space Conversion | O(n×m) | O(n×m) | Linear scan of all pixels |
| Convolution | O(n×m×k²) | O(n×m) | k = kernel size |
| Median Filter | O(n×m×k²×log(k²)) | O(k²) | Sorting overhead |
| DFT | O(n×m×log(n×m)) | O(n×m) | FFT algorithm |

### Filter Effectiveness Comparison

| **Filter Type** | **Noise Reduction** | **Edge Preservation** | **Computational Cost** |
|-----------------|-------------------|---------------------|----------------------|
| Average | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Gaussian | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Median | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Fourier Gaussian | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 🛠️ Technical Implementation

### Key Libraries Used:
- **OpenCV (cv2)**: Image I/O, color conversions, DFT operations
- **NumPy**: Numerical computations, array operations
- **Matplotlib**: Plotting and visualization support

### Implementation Highlights:
1. **Manual Implementation**: All algorithms implemented from scratch without using built-in OpenCV functions
2. **Pixel-level Processing**: Direct manipulation of pixel values for educational understanding
3. **Multiple Approaches**: Different methods for same operations to demonstrate various techniques
4. **Comprehensive Coverage**: Covers spatial and frequency domain processing

## 🤝 Contributing

This project is designed for educational purposes. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is created for educational purposes as part of CSE 573 coursework.

---

## 🎓 Academic Information

**Course**: CSE 573 - Computer Vision and Image Processing  
**Institution**: University at Buffalo, Department of Computer Science and Engineering  
**Instructor**: Prof. Nalini Ratha, PhD  

### Learning Objectives Achieved:
- ✅ Understanding of color space representations and conversions
- ✅ Implementation of spatial domain filtering techniques  
- ✅ Frequency domain analysis and filtering
- ✅ Practical experience with OpenCV and NumPy
- ✅ Manual implementation of core CV algorithms

---

*This project demonstrates fundamental computer vision concepts through hands-on implementation, providing a solid foundation for advanced CV topics.*