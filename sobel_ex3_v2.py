# sobel_ex3_v2 with display options execpt for open and median

import cv2
import numpy as np
from matplotlib import pyplot as plt

# testing options
outputs = False
display = True  # Sobel
display2 = True  # Vertices

# ---CODE---
img = cv2.imread(
    '/Users/diogo/Desktop/Cadeiras/Mestrado/labs II/scripts/Imagem/Meds/Images/benuron_from_net_40perCent.jpg',
    cv2.IMREAD_GRAYSCALE)

# -------------------------------------------------------------------------------------------------------------SOBEL
# https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php

if display:
    # histogram
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    plt.subplot(2, 1, 1), plt.plot(bin_edges[0:-1], histogram)
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0, 255])  # <- named arguments do not work here

# gaussian blur
img = cv2.GaussianBlur(img, (11, 11), 0) # todo ------------------------------------------------------calibrar

if display:
    # histogram
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    plt.subplot(2, 1, 2), plt.plot(bin_edges[0:-1], histogram)
    plt.title("Hist after gaussian blur")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0, 255])  # <- named arguments do not work here
    plt.show()

# laplace
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
# sobel
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobelCombined = cv2.bitwise_or(sobelX, sobelY)

titles = ['sobelX', 'sobelY', 'Laplacian', 'sobelCombined']
images = [sobelX, sobelY, lap, sobelCombined]

if display:
    # sobel, lap and hists
    plt.subplot(3, 4, 9), plt.imshow(img, 'gray')
    plt.title("image")
    plt.xticks([]), plt.yticks([])
    c = 1
    for i in range(len(images)):
        # images
        plt.subplot(3, 4, c), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        c += 1
        # hists
        # create the histogram
        histogram, bin_edges = np.histogram(images[i], bins=256, range=(0, 255))
        plt.subplot(3, 4, c), plt.plot(bin_edges[0:-1], histogram)
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixels")
        plt.xlim([0, 255])  # <- named arguments do not work here
        c += 1
    plt.show()


# Treshold
if display:
    plt.subplot(2, 3, 1), plt.imshow(img, 'gray')
    plt.title("image")
    plt.xticks([]), plt.yticks([])

threshImages = images
threshold = 30  # todo 10-200 ---------------------------------------------------------------calibrar
for i in range(len(images)):
    ret, threshold_image = cv2.threshold(images[i], threshold, 255, 0)
    threshImages[i] = threshold_image
    if display:
        plt.subplot(2, 3, i + 2), plt.imshow(threshold_image, 'gray')
        plt.title(titles[i] + " thresh " + str(threshold))
        plt.xticks([]), plt.yticks([])
if display:
    plt.show()

sobeilos = 1  # 1-open 2-median 3-both 4-nothing
if sobeilos == 1 or sobeilos == 3:
    ## limpar o sobel x e y antes de juntar
    Ks = 3;     kernel = np.ones((Ks, Ks), np.uint8) # todo ------------------------------------------------------calibrar
    Ks2 = 1;    kernel2 = np.ones((Ks2, Ks2), np.uint8) # todo ------------------------------------------------------calibrar
    c = 1
    sobels = [images[0], images[1], images[2]]

    for i in range(len(sobels)):
        plt.subplot(3, 3, c), plt.imshow(sobels[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        c += 1
        sobel_c = cv2.morphologyEx(sobels[i], cv2.MORPH_OPEN, kernel)
        # threshImages[i] = sobel_c         # ------------------------------------save?
        plt.subplot(3, 3, c), plt.imshow(sobel_c, 'gray')
        plt.title(titles[i] + " open k_" + str(Ks))
        plt.xticks([]), plt.yticks([])
        c += 1
        sobel_c = cv2.morphologyEx(sobels[i], cv2.MORPH_OPEN, kernel2)
        threshImages[i] = sobel_c  # ------------------------------------save?
        plt.subplot(3, 3, c), plt.imshow(sobel_c, 'gray')
        plt.title(titles[i] + " open k_" + str(Ks2))
        plt.xticks([]), plt.yticks([])
        c += 1

    plt.show()

elif sobeilos == 2 or sobeilos == 3:
    # median
    from skimage.filters.rank import median
    from skimage.morphology import disk

    ## limpar o sobel x e y antes de juntar
    kernel = np.ones((3, 3), np.uint8) # todo ------------------------------------------------------calibrar
    kernel2 = np.ones((1, 1), np.uint8) # todo ------------------------------------------------------calibrar
    c = 1
    sobels = [images[0], images[1]]

    for i in range(len(sobels)):
        plt.subplot(3, 3, c), plt.imshow(sobels[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
        c += 1
        sobel_c = cv2.medianBlur(sobels[i], 3)
        # threshImages[i] = sobel_c     # ------------------------------------save?
        plt.subplot(3, 3, c), plt.imshow(sobel_c, 'gray')
        plt.title(titles[i] + "median k_5")
        plt.xticks([]), plt.yticks([])
        c += 1
        sobel_c = cv2.medianBlur(sobels[i], 1)
        threshImages[i] = sobel_c  # ------------------------------------save?
        plt.subplot(3, 3, c), plt.imshow(sobel_c, 'gray')
        plt.title(titles[i] + "median k_3")
        plt.xticks([]), plt.yticks([])
        c += 1

    plt.show()


# close (dilate and erode)
kernel = np.ones((3, 3), np.uint8)  # todo ------------------------------------------------------calibrar
kernel2 = np.ones((11, 11), np.uint8)  # todo ------------------------------------------------------calibrar
closedImages = images
c = 1

if display:
    # original img
    plt.subplot(3, 4, 9), plt.imshow(img, 'gray')
    plt.title("image")
    plt.xticks([]), plt.yticks([])
# late combined sobel
sobelCombined = cv2.bitwise_or(threshImages[0], threshImages[1])

for i in range(len(threshImages)):
    closed = cv2.morphologyEx(threshImages[i], cv2.MORPH_CLOSE, kernel)
    # closedImages[i] = closed
    if display:
        plt.subplot(3, 4, c), plt.imshow(closed, 'gray')
        plt.title(titles[i] + " k_3")
        plt.xticks([]), plt.yticks([])
    c += 1
    closed2 = cv2.morphologyEx(threshImages[i], cv2.MORPH_CLOSE, kernel2)
    closedImages[i] = closed2
    if display:
        plt.subplot(3, 4, c), plt.imshow(closed2, 'gray')
        plt.title(titles[i] + " k_11")
        plt.xticks([]), plt.yticks([])
    c += 1

    sobelCombined = cv2.morphologyEx(sobelCombined, cv2.MORPH_CLOSE, kernel)
    if display:
        plt.subplot(3, 4, 12), plt.imshow(sobelCombined, 'gray')
        plt.title("LATE SOBEL")
        plt.xticks([]), plt.yticks([])
if display:
    plt.show()

if outputs & display:
    cv2.imshow("teste", sobelCombined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
