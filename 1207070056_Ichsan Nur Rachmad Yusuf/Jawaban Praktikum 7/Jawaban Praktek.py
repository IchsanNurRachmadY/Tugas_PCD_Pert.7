import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bunga-kamboja.png',0)#Kode ini membaca gambar dengan nama dan menyimpannya ke variabel img.

# menerapkan algoritma high-pass filtering:
# laplacian
laplacian = cv2.Laplacian(img,cv2.CV_64F)

# sobel dengan ukuran kernel 3
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

# perbesar ukuran hasil plotting
#Kode ini mengatur ukuran gambar plot menjadi (10, 10) menggunakan plt.rcParams.
plt.rcParams["figure.figsize"] = (10,10)

# menampilkan hasil filter
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])#Kode ini membuat subplot dan menampilkan gambar grayscale asli.
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])#Kode ini membuat subplot lainnya dan menampilkan gambar setelah menerapkan filter Laplacian.
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])#Kode ini membuat subplot lainnya dan menampilkan gambar setelah menerapkan filter Sobel pada sumbu x.
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])#Kode ini membuat subplot lainnya dan menampilkan gambar setelah menerapkan filter Sobel pada sumbu y.
plt.show()

# memanggil citra sebagai grayscale
img = cv2.imread('bunga-kamboja.png',0)#Kode ini menampilkan plot dengan semua subplot yang dibuat.

# memanggil fungsi Canny Edges dengan argument (citra, nilai_min, nilai_max)
edges = cv2.Canny(img,100,200)#Kode ini mengaplikasikan metode Canny untuk mendeteksi tepi pada gambar img. Argumen 100 dan 200 menentukan threshold bawah dan atas untuk proses deteksi tepi.

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])#Kode ini membuat subplot pertama dan menampilkan gambar asli.
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])#Kode ini membuat subplot kedua dan menampilkan gambar hasil deteksi tepi.

plt.show()#Kode ini menampilkan plot dengan kedua subplot yang telah dibuat.

img = cv2.imread('bunga-kamboja.png',0)#Kode ini membaca gambar dan menyimpannya ke dalam variabel img.

# Hitungan threshold.
# Perhatikan nilai ambang batas bawah dan atas dari tiap fungsi
# yang diberikan
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)#Mengubah piksel dengan intensitas di atas ambang batas menjadi nilai maksimum (dalam hal ini 255) dan di bawah ambang batas menjadi nilai nol.
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)#mengubah piksel dengan intensitas di atas ambang batas menjadi nilai nol dan di bawah ambang batas menjadi nilai maksimum.
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)#Memotong (truncates) piksel yang memiliki intensitas di atas ambang batas menjadi ambang batas itu sendiri, sementara piksel dengan intensitas di bawah ambang batas tetap tidak berubah.
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)#Mengubah piksel dengan intensitas di atas ambang batas menjadi nilai intensitas itu sendiri, sedangkan piksel dengan intensitas di bawah ambang batas tetap tidak berubah.
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)#mengubah piksel dengan intensitas di atas ambang batas menjadi nilai nol, sementara piksel dengan intensitas di bawah ambang batas tetap tidak berubah.

# menampilkan hasil
titles = ['Gambar asli','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']#udul-judul disimpan dalam list titles,
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]#gambar-gambar (termasuk gambar asli dan hasil thresholding) disimpan dalam list images.

# menampilkan beberapa gambar sekaligus
for i in range(6):
    plt.subplot(3,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()#Kode ini menampilkan plot dengan semua subplot yang telah dibuat.

img = cv2.medianBlur(img,7)

# Binary Threshold
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Adaptive Threshold dengan Mean
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)

# Adaptive Threshold dengan Gaussian
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)


# Plotting
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# menampilkan hasil
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()#Kode ini menampilkan plot dengan semua subplot yang telah dibuat.