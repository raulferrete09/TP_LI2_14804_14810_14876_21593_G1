"""
implementation of code from meds_imagePrep_v2_all.py
limpo (só usando um processo e sem funçoes dentro?

1-sobel_ex3_v2, 2-vertices_ex3_juncao_do_1_e_2V2, 3-transform_ex1, 4-OCR from labs2_med

"""
# todo 1 calibrar e limpar o sobel_ex3 aqui para fazer só um processo         !!!!!!!!!!!!!!!!!!!!!!!!!!!!
# todo 2 calibrar e limpar o os kernels para fazer só um processo         !!!!!!!!!!!!!!!!!!!!!!!!!!!!

import cv2
import numpy as np
from matplotlib import pyplot as plt
import math as m
from pytesseract import pytesseract
from pytesseract import Output

testing = True


def ocrOut(img):
    # testing options
    all = False
    finalOutput = all and False
    outputs = all or True
    display = all or True # Sobel
    display2 = all or False  # Vertices
    display3 = all or False  # Transformation
    display4 = all or True  # OCR
    # transform
    aspectRatio = True

    # ---CODE---
    if len(img.shape) > 2:  # se nao vier em grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgOriginal = img  # FOR FINAL TRANSFORM

    # -----------------------------------------------------------------------------------------------------  SOBEL

    if display:
        # histogram
        histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))

        plt.subplot(1, 2, 1)
        plt.imshow(img, 'gray')
        plt.title('Imagem Original (1)')
        plt.xlim(100, 200)
        plt.ylim(200, 100)

    # gaussian blur
    img = cv2.GaussianBlur(img, (11, 11), 0)  # todo ------------------------------------------------------calibrar

    if display:
        # histogram
        histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))

        plt.subplot(1, 2, 2)
        plt.imshow(img, 'gray')
        plt.title('Imagem Filtrada (2)')
        plt.xlim(100,200)
        plt.ylim(200, 100)
        plt.show()

    # sobel
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    titles = ['sobelX', 'sobelY']
    images = [sobelX, sobelY]

    if display:
        # sobel, lap and hists
        plt.subplot(2, 3, 1), plt.imshow(img, 'gray')
        plt.title("image")
        plt.xticks([]), plt.yticks([])
        c = 2
        for i in range(len(images)):
            # images
            plt.subplot(2, 3, c), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
            c += 1
            # hists
            # create the histogram
            histogram, bin_edges = np.histogram(images[i], bins=256, range=(0, 255))
            plt.subplot(2, 3, c), plt.plot(bin_edges[0:-1], histogram)
            plt.title("Grayscale Histogram")
            plt.xlabel("grayscale value")
            plt.ylabel("pixels")
            plt.xlim([0, 255])  # <- named arguments do not work here
            c += 2
        plt.show()

    # Treshold
    if display:
        plt.subplot(1, 3, 1), plt.imshow(img, 'gray')
        plt.title("image")
        plt.xticks([]), plt.yticks([])

    threshImages = images
    threshold = 30  # todo 10-200 ---------------------------------------------------------------calibrar
    for i in range(len(images)):
        ret, threshold_image = cv2.threshold(images[i], threshold, 255, 0)
        threshImages[i] = threshold_image
        if display:
            plt.subplot(1, 3, i + 2), plt.imshow(threshold_image, 'gray')
            plt.title(titles[i] + " thresh " + str(threshold))
            plt.xticks([]), plt.yticks([])
    if display:
        plt.show()

    ## limpar o sobel x e y antes de juntar
    Ks = 3;
    kernel = np.ones((Ks, Ks), np.uint8)  # todo ------------------------------------------------------calibrar
    Ks2 = 1;
    kernel2 = np.ones((Ks2, Ks2), np.uint8)  # todo ------------------------------------------------------calibrar
    c = 1
    sobels = [images[0], images[1]]

    for i in range(len(sobels)):
        if display:
            plt.subplot(2, 3, c), plt.imshow(sobels[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
            c += 1
            sobel_c = cv2.morphologyEx(sobels[i], cv2.MORPH_OPEN, kernel)
            # threshImages[i] = sobel_c         # ------------------------------------save?
            plt.subplot(2, 3, c), plt.imshow(sobel_c, 'gray')
            plt.title(titles[i] + " open k_" + str(Ks))
            plt.xticks([]), plt.yticks([])
            c += 1
            sobel_c = cv2.morphologyEx(sobels[i], cv2.MORPH_OPEN, kernel2)
            threshImages[i] = sobel_c  # ------------------------------------save?
            plt.subplot(2, 3, c), plt.imshow(sobel_c, 'gray')
            plt.title(titles[i] + " open k_" + str(Ks2))
            plt.xticks([]), plt.yticks([])
            c += 1
    if display:
        plt.show()

    # close (dilate and erode)
    Ks = 3;
    kernel = np.ones((Ks, Ks), np.uint8)  # todo ------------------------------------------------------calibrar
    Ks2 = 11;
    kernel2 = np.ones((Ks2, Ks2), np.uint8)  # todo ------------------------------------------------------calibrar

    # late combined sobel
    sobelCombined = cv2.bitwise_or(threshImages[0], threshImages[1])
    sobelCombined2 = cv2.morphologyEx(sobelCombined, cv2.MORPH_CLOSE, kernel2)
    sobelCombined = cv2.morphologyEx(sobelCombined, cv2.MORPH_CLOSE, kernel)

    if display:
        # original img
        plt.subplot(1, 3, 1), plt.imshow(img, 'gray')
        plt.title("image")
        plt.xticks([]), plt.yticks([])

        plt.subplot(1, 3, 2), plt.imshow(sobelCombined, 'gray')
        plt.title("SOBEL Magnitude k_" + str(Ks))
        plt.xticks([]), plt.yticks([])
        plt.subplot(1, 3, 3), plt.imshow(sobelCombined2, 'gray')
        plt.title("SOBEL Magnitude k_" + str(Ks2))
        plt.xticks([]), plt.yticks([])
        plt.show()

    if outputs or display:
        cv2.imshow("SOBEL OUTPUT", sobelCombined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # -------------------------------------------------------------------------------------------------------------VERTICES

    # variable connection
    img = cv2.cvtColor(sobelCombined, cv2.COLOR_GRAY2BGR)
    gray = sobelCombined

    ## FIRST CONTOURS
    # find contours in the edged image, keep only the largest ones, and initialize our screen contour
    cnts = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # (*1) (*3)
    cnts = cnts[0]  # fot open cv2	(*2)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # (*4) (*3)

    # loop and represent all contours
    for c in cnts:
        # approximate the contour  # (*3) (*1)
        perimeter = 0.0001 * cv2.arcLength(c, True)  # o multiplicador deve ser baixo pra incluir todos os contours
        approx = cv2.approxPolyDP(c, perimeter, True)

        # prencher todos os objectos
        gray = cv2.drawContours(gray, [approx], -1, (255, 255, 255), -1)
        img = cv2.drawContours(img, [approx], -1, (0, 255, 0), -1)

    if display2:
        plt.subplot(2, 2, 1), plt.imshow(gray, 'gray')
        plt.title("caixa e vertices")
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 2), plt.imshow(img)
        plt.title("caixa e vertices rgb")
        plt.xticks([]), plt.yticks([])

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # MORPH CLOSE
    Ks = 3;
    kernel = np.ones((Ks, Ks), np.uint8)  # todo ------------------------------------------------------calibrar
    Ks2 = 20;
    kernel2 = np.ones((Ks2, Ks2), np.uint8)  # todo ------------------------------------------------------calibrar
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    closed2 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)

    if display2:
        plt.subplot(2, 2, 3), plt.imshow(closed, 'gray')
        plt.title("closed k_" + str(Ks))
        plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 4), plt.imshow(closed2, 'gray')
        plt.title("closed k_" + str(Ks2))
        plt.xticks([]), plt.yticks([])

        plt.show()

    gray = closed2

    # SECOND CONTOURS
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(gray.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # (*1) (*3)
    cnts = cnts[0]  # fot open cv2	(*2)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]  # (*4) (*3)
    vertices = None

    # loop over our contours
    maxArea = 0
    for c in cnts:
        # approximate the contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if maxArea < cv2.contourArea(c):
            maxArea = cv2.contourArea(c)
            vertices = box

        cv2.drawContours(img, [box], 0, (0, 255, 255), 2)

    cv2.drawContours(img, [vertices], -1, (0, 0, 255), 3)  # perimetro (*3)
    for i in range(len(vertices)):
        cv2.circle(img, (vertices[i][0],vertices[i][1]), 3, (0, 255, 0), 3)  # vertices
    cv2.circle(img, (vertices[i][0],vertices[i][1]), 2, (255, 0, 0), 3)  # vertice independente

    # identifacação dos cantos para usar no transform (variable connection)
    max1 = [[0, 0]]
    max2 = [[0, 0]]
    min1 = [[999, 999]]
    min2 = [[999, 999]]

    if display2:
        print("array :", vertices)

    for i in range(4):
        if max1[0][0] < vertices[i][0]:
            max1[0][0] = vertices[i][0]
            max1[0][1] = vertices[i][1]
        if min1[0][0] > vertices[i][0]:
            min1[0][0] = vertices[i][0]
            min1[0][1] = vertices[i][1]

    for i in range(4):
        if (max2[0][0] < vertices[i][0]) & (vertices[i][1] != max1[0][1]):
            max2[0][0] = vertices[i][0]
            max2[0][1] = vertices[i][1]
        if (min2[0][0] > vertices[i][0]) & (vertices[i][1] != min1[0][1]):
            min2[0][0] = vertices[i][0]
            min2[0][1] = vertices[i][1]

    # DESNECESSÁRIO
    if display2:
        print(max1, max2, min1, min2)

    if max1[0][1] > max2[0][1]:
        br = max1
        tr = max2
    else:
        br = max2
        tr = max1
    if min1[0][1] > min2[0][1]:
        bl = min1
        tl = min2
    else:
        bl = min2
        tl = min1

    if outputs or display2:
        showCorner = [tl, bl, br, tr]
        cornerTxt = ["TL", "BL", "BR", "TR"]
        for i in range(len(showCorner)):
            print(showCorner[i][0])
            m = cv2.circle(img, showCorner[i][0], 2, (255 * (5 - (i + 1)) / 5, 0, 255 * (i + 1) / 5),
                           10)  # vertice independente
            cv2.putText(img, cornerTxt[i], showCorner[i][0], cv2.FONT_HERSHEY_PLAIN, 2,
                        (255 * (5 - (i + 1)) / 5, 0, 255 * (i + 1) / 5), 2)
            cv2.imshow("Bounding box da caixa e vertices", img)
            cv2.waitKey(0)

    # -----------------------------------------------------------------------------------------------------  TRANSFORMATION

    # WORKS to unwarp or transform a rectangular object in an image if given the vertices !!!!

    def unwarp(img, src, dst, newSize, testing):
        # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
        M = cv2.getPerspectiveTransform(src, dst)
        # use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(img, M, newSize, flags=cv2.INTER_LINEAR)

        if testing:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            f.subplots_adjust(hspace=.2, wspace=.05)
            ax1.imshow(imgOriginal, 'gray')
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(warped, 'gray')
            ax2.set_title('Unwarped Image', fontsize=30)
            plt.show()

        return warped, M

    h, w = imgOriginal.shape[:2]
    tl = tl[0]
    tr = tr[0]
    bl = bl[0]
    br = br[0]

    if display3:
        print(tl, bl, br, tr)

    ##  CODIGO para obter uma transformação mantendo o aspectRatio do objeto rectangular
    if aspectRatio:
        if display3:
            print("Aspect Ratio Mode: ON !!!!")
        # get object height and width
        leftSize = int(((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2) ** .5);  # print(leftSize)
        rightSize = int(((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2) ** .5);  # print(rightSize)
        topSize = int(((tl[0] - tr[0]) ** 2 + (tl[1] - tr[1]) ** 2) ** .5);  # print(topSize)
        bottomSize = int(((bl[0] - br[0]) ** 2 + (bl[1] - br[1]) ** 2) ** .5);  # print(bottomSize)
        # salva a maior largura e comprimento
        h = leftSize if leftSize > rightSize else rightSize
        w = topSize if topSize > bottomSize else bottomSize

    src = np.float32([(bl[0], bl[1]),  # bottom left
                      (tl[0], tl[1]),  # top left
                      (br[0], br[1]),  # bottom right
                      (tr[0], tr[1])])  # top right

    if display3:
        print("output height: ", h, "\twidth:", w)

    dst = np.float32([(0, h), (0, 0), (w, h), (w, 0)])

    [img, m] = unwarp(imgOriginal, src, dst, (w, h), display3)

    # delete in implementation !!!!!
    if outputs or display3:
        cv2.imshow("Image for ocr", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #  ------------------------------------------------------------------------------------------------------- TESSERACT OCR

    # OCR
    image_data = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Printing each word
    if display4 or finalOutput:  # SEND THE ARRAY OF WORDS
        print(image_data['text'])
        for word in image_data['text']:
            print(word)

    for i, word in enumerate(image_data['text']):
        if word != '':
            x, y, w, h = image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][
                i]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(img, word, (x, y - 16), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    if outputs or display4:
        cv2.imshow("OCR IMAGE", img)
        cv2.waitKey(0)

    return image_data['text']

if testing:
    img = cv2.imread(
        '/Users/diogo/Desktop/Cadeiras/Mestrado/labs II/scripts/Imagem/Meds/Images/benuron_from_net_40perCent.jpg',
        cv2.IMREAD_GRAYSCALE)

print(ocrOut(img))
