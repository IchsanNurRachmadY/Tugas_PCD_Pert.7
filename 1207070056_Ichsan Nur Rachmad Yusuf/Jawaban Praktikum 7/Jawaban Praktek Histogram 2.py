import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lotus.jpg', 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

plt.rcParams["figure.figsize"] = (12, 8)

fig, axs = plt.subplots(2, 3)

axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 0].axis('off')

axs[1, 0].hist(img.ravel(), 256, [0, 256], color='blue')
axs[1, 0].set_title('Histogram - Original')

axs[0, 1].imshow(laplacian, cmap='gray')
axs[0, 1].set_title('Laplacian')
axs[0, 1].axis('off')

axs[1, 1].hist(laplacian.ravel(), 256, [0, 256], color='blue')
axs[1, 1].set_title('Histogram - Laplacian')

axs[0, 2].imshow(sobelx, cmap='gray')
axs[0, 2].set_title('Sobel X')
axs[0, 2].axis('off')

axs[1, 2].hist(sobelx.ravel(), 256, [0, 256], color='blue')
axs[1, 2].set_title('Histogram - Sobel X')

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()