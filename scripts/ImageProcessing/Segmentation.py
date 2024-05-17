import cv2
import matplotlib.pyplot as plt

# Read the image
# image = cv2.imread('Input/image_0.png')
# image = cv2.imread('Input/image_310.png')
# image = cv2.imread('Input/image_557.png')
# image = cv2.imread('Input/image_764.png')
image = cv2.imread('Input/image_815.png')
# image = cv2.imread('Input/image_894.png')
# image = cv2.imread('Input/image_1047.png')
# image = cv2.imread('Input/image_1199.png')
# image = cv2.imread('Input/image_1274.png')
# image = cv2.imread('Input/image_1581.png')
# image = cv2.imread('Input/image_4003.png')
# image = cv2.imread('Input/image_4467.png')
# image = cv2.imread('Input/image_4526.png')

# Convert the image to different color spaces
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
image_luv = cv2.cvtColor(image, cv2.COLOR_BGR2Luv)
image_xyz = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)

# Display the images
plt.figure(1, figsize=(15, 15))

plt.subplot(3, 3, 1)
plt.imshow(image)
plt.title('Original')

plt.subplot(3, 3, 2)
plt.imshow(image_gray, cmap='gray')
plt.title('Grayscale')

plt.subplot(3, 3, 3)
plt.imshow(image_hsv)
plt.title('HSV')

plt.subplot(3, 3, 4)
plt.imshow(image_rgb)
plt.title('RGB')

plt.subplot(3, 3, 5)
plt.imshow(image_lab)
plt.title('LAB')

plt.subplot(3, 3, 6)
plt.imshow(image_hls)
plt.title('HLS')

plt.subplot(3, 3, 7)
plt.imshow(image_luv)
plt.title('LUV')

plt.subplot(3, 3, 8)
plt.imshow(image_xyz)
plt.title('XYZ')

plt.tight_layout()
plt.show()

# Split the images into separate components
image_hsv_h, image_hsv_s, image_hsv_v = cv2.split(image_hsv.astype(float))
image_rgb_r, image_rgb_g, image_rgb_b = cv2.split(image_rgb.astype(float))
image_lab_l, image_lab_a, image_lab_b = cv2.split(image_lab.astype(float))
image_hls_h, image_hls_l, image_hls_s = cv2.split(image_hls.astype(float))
image_luv_l, image_luv_u, image_luv_v = cv2.split(image_luv.astype(float))
image_xyz_x, image_xyz_y, image_xyz_z = cv2.split(image_xyz.astype(float))

# Display the images
plt.figure(2, figsize=(24, 24))

plt.subplot(6, 3, 1)
plt.imshow(image_hsv_h)
plt.title('HSV H')

plt.subplot(6, 3, 2)
plt.imshow(image_hsv_s)
plt.title('HSV S')

plt.subplot(6, 3, 3)
# plt.imshow((200 < image_hsv_v < 250).astype('uint8')*255)
plt.imshow(cv2.inRange(image_hsv_v, 250, 255))
plt.title('HSV V')

plt.subplot(6, 3, 4)
plt.imshow(image_rgb_r)
plt.title('RGB R')

plt.subplot(6, 3, 5)
plt.imshow(image_rgb_g)
plt.title('RGB G')

plt.subplot(6, 3, 6)
plt.imshow(image_rgb_b)
plt.title('RGB B')

plt.subplot(6, 3, 7)
plt.imshow(image_lab_l)
plt.title('LAB L')

plt.subplot(6, 3, 8)
plt.imshow(image_lab_a)
plt.title('LAB A')

plt.subplot(6, 3, 9)
plt.imshow(image_lab_b)
plt.title('LAB B')

plt.subplot(6, 3, 10)
plt.imshow(image_hls_h)
plt.title('HLS H')

plt.subplot(6, 3, 11)
plt.imshow(image_hls_l)
plt.title('HLS L')

plt.subplot(6, 3, 12)
plt.imshow(image_hls_s)
plt.title('HLS S')

plt.subplot(6, 3, 13)
plt.imshow(image_luv_l)
plt.title('LUV L')

plt.subplot(6, 3, 14)
plt.imshow(image_luv_u)
plt.title('LUV U')

plt.subplot(6, 3, 15)
plt.imshow(image_luv_v)
plt.title('LUV V')

plt.subplot(6, 3, 16)
plt.imshow(image_xyz_x)
plt.title('XYZ X')

plt.subplot(6, 3, 17)
plt.imshow(image_xyz_y)
plt.title('XYZ Y')

plt.subplot(6, 3, 18)
plt.imshow(image_xyz_z)
plt.title('XYZ Z')

plt.tight_layout()
plt.show()

# Perform tophat operation on image_luv_u
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
image_tophat_luv_u = cv2.morphologyEx(image_luv_u, cv2.MORPH_TOPHAT, kernel)

plt.figure(3, figsize=(15, 15))

plt.subplot(2, 1, 1)
plt.imshow(image_tophat_luv_u, cmap='gray')
plt.title('Tophat LUV U')

plt.subplot(2, 1, 2)
plt.imshow(image_rgb)
plt.title('Original')

plt.show()

# Perform tophat operation on image_hls_s
# image_hls_s = cv2.GaussianBlur(image_hls_s, (7, 7), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
image_tophat_hls_s = cv2.morphologyEx(image_hls_s, cv2.MORPH_TOPHAT, kernel)

plt.figure(4, figsize=(15, 15))

plt.subplot(2, 1, 1)
plt.imshow(image_tophat_hls_s, cmap='gray')
plt.title('Tophat HLS S')

plt.subplot(2, 1, 2)
plt.imshow(image_rgb)
plt.title('Original')

plt.show()

# Perform tophat operation on image_hls_s
# image_lab_b = cv2.GaussianBlur(image_lab_b, (7, 7), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
image_tophat_lab_b = cv2.morphologyEx(image_lab_b, cv2.MORPH_TOPHAT, kernel)

plt.figure(5, figsize=(15, 15))

plt.subplot(2, 1, 1)
plt.imshow(image_tophat_lab_b, cmap='gray')
plt.title('Tophat LAB B')

plt.subplot(2, 1, 2)
plt.imshow(image_rgb)
plt.title('Original')

plt.show()

# Multiply the tophat images
image_tophat_mult = cv2.multiply(image_tophat_hls_s, image_tophat_lab_b)

plt.figure(6, figsize=(15, 15))

plt.subplot(2, 1, 1)
plt.imshow(image_tophat_mult, cmap='gray')
plt.title('Tophat Multiplication')

plt.subplot(2, 1, 2)
plt.imshow(image_rgb)
plt.title('Original')

plt.show()

# Multiply the images and apply tophat
image_tophat_mult = cv2.multiply(image_hls_s, image_lab_b)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
image_tophat_mult = cv2.morphologyEx(image_tophat_mult, cv2.MORPH_TOPHAT, kernel)

plt.figure(7, figsize=(15, 15))

plt.subplot(2, 1, 1)
plt.imshow(image_tophat_mult, cmap='gray')
plt.title('Multiplication and Tophat')

plt.subplot(2, 1, 2)
plt.imshow(image_rgb)
plt.title('Original')

plt.show()