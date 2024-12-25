import cv2
import numpy as np
import matplotlib.pyplot as plt
# PART A
bgr_image = cv2.imread("Lenna.png")
rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
normalized_img = rgb_img / 255.0
height, width, channels = rgb_img.shape
hsv_img_1 = np.zeros((height, width, 3), dtype=np.float32)
for i in range(height):
    for j in range(width):
        r, g, b = normalized_img[i, j]
        v = max(r, g, b)
        if v == 0:
            s = 0
        else:
            s = (v - min(r, g, b)) / v
        if v == r:
            h = 60 * (g - b) / (v - min(r, g, b))
        elif v == g:
            h = 120 + 60 * (b - r) / (v - min(r, g, b))
        elif v == b:
            h = 240 + 60 * (r - g) / (v - min(r, g, b))
        elif r==g==b:
            h=0
        if h < 0:
            h = h + 360
        v = 255 * v
        s = 255 * s
        h = h / 2
        hsv_img_1[i, j] = [h, s, v]
cv2.imwrite("hsv_image_1.png", hsv_img_1)
hsv = cv2.imread("hsv_image_1.png")
cv2.imshow("HSV color space", hsv)
cv2.waitKey(0)
# same operation done with the formula used in class
bgr_image = cv2.imread("Lenna.png")
rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
normalized_img = rgb_img / 255.0
hsv_img_2 = np.zeros((height, width, 3), dtype=np.float64)
for i in range(height):
    for j in range(width):
        r, g, b = normalized_img[i, j]
        v = (r + g + b) / 3
        s = 1 - ((3/(r + g + b)) * (min(r, g, b)))
        my_theta = np.arccos((0.5 * ((r - g) + (r - b))) /
                                     (np.sqrt(((np.square(r - g)) + ((r - b) * (g - b))))))
        if b <= g:
            h = my_theta
        elif b > g:
            h = 360 - my_theta
        v = 255 * v
        s = 255 * s
        h = h / 2
        hsv_img_2[i, j] = [h, s, v]
cv2.imwrite("hsv_image_2.png", hsv_img_2)
hsv_2 = cv2.imread("hsv_image_2.png")
cv2.imshow("HSV color space (class)", hsv_2)
cv2.waitKey(0)
# RGB to CMYK
bgr_image = cv2.imread("Lenna.png")
rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
normalized_img = rgb_img / 255.0
height, width, _ = rgb_img.shape
cmyk_img = np.zeros((height, width, 4), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        r, g, b = normalized_img[i, j]
        c = 1 - r
        m = 1 - g
        y = 1 - b
        k = min(c, m, y)
        if k == 1:
            c, m, y = 0, 0, 0
        else:
            c = (c - k) / (1 - k)
            m = (m - k) / (1 - k)
            y = (y - k) / (1 - k)
        cmyk_img[i, j] = [c * 255, m * 255, y * 255, k * 255]
cv2.imwrite("cmyk_image.png", cv2.cvtColor(cmyk_img, cv2.COLOR_BGRA2BGR))
cmyk_img = cv2.imread("cmyk_image.png")
cv2.imshow("cmyk image stored as BGRA", cmyk_img)
cv2.waitKey(0)
# RGB to LAB
bgr_image = cv2.imread("Lenna.png")
rgb_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
def f(t):
    if t > 0.008856:
        return t ** (1 / 3)
    elif t <= 0.008856:
        return 7.787 * t + 16 / 116
normalized_img = rgb_img / 255.0
lab_img = np.zeros((height, width, 3), dtype=np.float64)
for i in range(height):
    for j in range(width):
        r, g, b = normalized_img[i, j]
        x = (0.412453 * r) + (0.357580 * g) + (0.180423 * b)
        y = (0.212671 * r) + (0.715160 * g) + (0.072169 * b)
        z = (0.019334 * r) + (0.119193 * g) + (0.950227 * b)
        x = x / 0.950456
        z = z / 1.088754
        if y > 0.008856:
            L = 116 * (y ** (1/3)) - 16
        elif y <= 0.008856:
            L = 903.3 * y
        A = 500 * (f(x) - f(y)) + 128
        B = 200 * (f(y) - f(z)) + 128
        L = L * (255 / 100)
        lab_img[i, j] = [L, A, B]
cv2.imwrite("lab_image.png", lab_img)
lab_img = cv2.imread("lab_image.png")
cv2.imshow("LAB image", lab_img)
cv2.waitKey(0)
# PART B
# CONVOLUTION
img = cv2.imread("Noisy_image.png", cv2.IMREAD_GRAYSCALE)
a = 1 / 9
filter = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.float32) * a
r, c = filter.shape
flip_k = np.zeros_like(filter)
for i in range(r):
    for j in range(c):
        flip_k[i, j] = filter[r - 1 - i, c - 1 - j]
my_c_filter = flip_k
i_h, i_w = img.shape
f_h, f_w = my_c_filter.shape
pad_height = f_h // 2
pad_width = f_w // 2
convolved_image = np.zeros_like(img, dtype=np.uint8)
for i in range(pad_height, i_h - pad_height):
    for j in range(pad_width, i_w - pad_width):
            region = img[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            conv = int(np.sum(region * my_c_filter))
            convolved_image[i, j] = conv
cv2.imwrite('convolved_image.png', convolved_image)
c_img = cv2.imread("convolved_image.png")
cv2.imshow("Convolved image", c_img)
cv2.waitKey(0)
# AVERAGING FILTER
height, width = img.shape
a = 1 / 9
my_filter = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=np.float32) * a
f_size = 3
average_image = np.zeros((height, width), dtype=np.uint8)
for y in range(f_size // 2, height - f_size // 2):
    for x in range(f_size // 2, width - f_size // 2):
            neighborhood = img[y - f_size // 2:y + f_size // 2 + 1,
                           x - f_size // 2:x + f_size // 2 + 1]
            avg_value = np.mean(neighborhood)
            average_image[y, x] = avg_value
cv2.imwrite('average_image.png', average_image)
a_img = cv2.imread("average_image.png")
cv2.imshow("Averaged image", a_img)
cv2.waitKey(0)
# Gaussian filtering
a = 1/16
g_kernel = a * np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])
k_size = g_kernel.shape[0]
op_img = np.zeros_like(img)
for i in range(height - k_size + 1):
    for j in range(width - k_size + 1):
        roi = img[i:i + k_size, j:j + k_size]
        filtered_value = np.sum(roi * g_kernel)
        op_img[i + k_size//2, j + k_size//2] = filtered_value
Gauss_img = op_img.astype(np.uint8)
cv2.imwrite('gaussian_image.jpg', Gauss_img)
g_img = cv2.imread('gaussian_image.jpg')
cv2.imshow("Gaussian filtered image", g_img)
cv2.waitKey(0)
# Median Filtering
m_kernel_size = 5
m_op_img = np.copy(img)
for i in range(2, height - 2):
    for j in range(2, width - 2):
        neighborhood = img[i - 2:i + 3, j - 2:j + 3]
        median_value = np.median(neighborhood, axis=None)
        m_op_img[i + m_kernel_size // 2, j + m_kernel_size // 2] = median_value
cv2.imwrite('median_image.jpg', m_op_img)
m_img = cv2.imread('median_image.jpg')
cv2.imshow("Median filtered image", m_img)
cv2.waitKey(0)
# Contrast and Brightness Adjustment
orig_image = cv2.imread('Uexposed.png')
contrast = 10.0
brightness = 50
adjusted_img = orig_image.copy()
for y in range(orig_image.shape[0]):
    for x in range(orig_image.shape[1]):
        for c in range(3):
            new_pixel_value = int((orig_image[y, x, c] * contrast) + brightness)
            adjusted_img[y, x, c] = max(0, min(255, new_pixel_value))
cv2.imwrite('adjusted_image.png', adjusted_img)
bright_img = cv2.imread('adjusted_image.png')
cv2.imshow("Improved Contrast and Brightness image", bright_img)
cv2.waitKey(0)
#PART C
img = cv2.imread("Noisy_image.png", cv2.IMREAD_GRAYSCALE)
height, width = img.shape
g_arr = img.astype(np.float32)
f_trans = cv2.dft(g_arr, flags=cv2.DFT_COMPLEX_OUTPUT)
f_trans_shift = np.fft.fftshift(f_trans)
my_magnitude = cv2.magnitude(f_trans_shift[:, :, 0], f_trans_shift[:, :, 1])
new_magnitude = np.log(my_magnitude + 1)
min_magnitude = np.min(new_magnitude)
max_magnitude = np.max(new_magnitude)
normalized_magnitude = new_magnitude / max_magnitude
scaled_magnitude = np.uint8(255 * normalized_magnitude)
cv2.imwrite('converted_fourier.png', scaled_magnitude)
cv2.imshow('Fourier Domain Image', scaled_magnitude)
cv2.waitKey(0)

# second part of PART C
def my_customised_g_filter(k_size, sigma):
    k = np.zeros((k_size, k_size), dtype=np.float32)
    c_o_k = k_size // 2
    for i in range(k_size):
        for j in range(k_size):
            x = i - c_o_k
            y = j - c_o_k
            k[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    k = k / k.sum()
    return k
size_of_filter = 75
std_deviation = 100.0
gaussian_kernel = my_customised_g_filter(size_of_filter, std_deviation)
new_g_kernel = np.stack((gaussian_kernel, np.zeros_like(gaussian_kernel)), axis=-1)
my_height_diff = f_trans_shift.shape[0] - new_g_kernel.shape[0]
my_width_diff = f_trans_shift.shape[1] - new_g_kernel.shape[1]
new_s_o_f = size_of_filter + 1
new_gaussian_kernel = my_customised_g_filter(new_s_o_f, std_deviation)
padded_gaussian_kernel = np.pad(new_gaussian_kernel, ((my_height_diff//2, my_height_diff//2), (my_width_diff//2, my_width_diff//2)), mode='constant')
padded_gaussian_kernel = padded_gaussian_kernel[:f_trans_shift.shape[0], :f_trans_shift.shape[1]]
filter_applied_f_trans_shift = f_trans_shift * padded_gaussian_kernel[:, :, np.newaxis]
f_image_shift = cv2.idft(np.fft.ifftshift(filter_applied_f_trans_shift), flags=cv2.DFT_SCALE)
mag_apply = cv2.magnitude(f_image_shift[:, :, 0], f_image_shift[:, :, 1])
log_apply = np.log(mag_apply + 1)
min_magnitude = np.min(log_apply)
max_magnitude = np.max(log_apply)
normalized_points = (log_apply - min_magnitude) / (max_magnitude - min_magnitude)
scaled_points = (normalized_points * 255).astype(np.uint8)
cv2.imshow('Original Image', img)
cv2.imshow('Filtered Image', scaled_points)
cv2.imwrite('gaussian_fourier.png', scaled_points)
cv2.waitKey(0)
