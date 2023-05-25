import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lotus.jpg')

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

kernel = np.ones((3, 5), np.float32) * 0.04
print(kernel)

filtered_img = cv2.filter2D(img, -1, kernel)

plt.rcParams["figure.figsize"] = (15, 15)

plt.subplot(221), plt.imshow(rgb_img), plt.title('Original')
plt.xticks([]), plt.yticks([])

hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.subplot(222), plt.plot(hist_img, color='r')
plt.title('Histogram Original')
plt.xlim([0, 256])

filtered_rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)

plt.subplot(223), plt.imshow(filtered_rgb_img), plt.title('Averaging')
plt.xticks([]), plt.yticks([])

hist_filtered_img = cv2.calcHist([filtered_img], [0], None, [256], [0, 256])

plt.subplot(224), plt.plot(hist_filtered_img, color='r')
plt.title('Histogram Filtered')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()