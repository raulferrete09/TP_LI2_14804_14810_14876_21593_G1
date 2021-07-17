
# código baseado no preenchimento de imagem pelo processo do vertices_ex1_V2.py
# e com a deteção de rectangulos baseada no vertices_ex2_rotated_BBpoints.py

import cv2
import numpy as np
from matplotlib import pyplot as plt

#options
outputs=True
display=True

#code
img = cv2.imread(
    "/Users/diogo/Desktop/Cadeiras/Mestrado/labs II/scripts/Imagem/Meds/Images/benuron_from_net_sobel_open_closed_K5.jpg",
)  # cv2.IMREAD_GRAYSCALE)

if display:
    cv2.imshow("imput img", img)
    cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = img

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


#todo eliminate import and plots (only for test)
if display:
    plt.subplot(2, 2, 1), plt.imshow(gray, 'gray')
    plt.title("caixa e vertices")
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(img)
    plt.title("caixa e vertices rgb")
    plt.xticks([]), plt.yticks([])

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# MORPH CLOSE
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((20, 20), np.uint8)  #todo in implementation the value probably should be less
closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
closed2 = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel2)

if display:
    plt.subplot(2, 2, 3), plt.imshow(closed, 'gray')
    plt.title("closed k_3")
    plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(closed2, 'gray')
    plt.title("closed k_11")
    plt.xticks([]), plt.yticks([])

    plt.show()
#todo--------------------------------------------

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
        print(vertices)
    #

    cv2.drawContours(img, [box], 0, (0, 255, 255), 2)

cv2.drawContours(img, [vertices], -1, (0, 0, 255), 3)  # perimetro (*3)
for i in range(len(vertices)):
    cv2.circle(img, vertices[i], 3, (0, 255,0), 3)  # vertices
cv2.circle(img, vertices[i], 2, (255, 0,0), 3)   # vertice independente
# ordem dos vertices é tr, tl, bl, br
if outputs or display:
    cv2.imshow("Bounding box da caixa e vertices", img)
    cv2.waitKey(0)


#todo identificar os vertices---------------------------------------------------------------------------------
# codigo raul
max1 = [[0, 0]]
max2 = [[0, 0]]
min1 = [[999, 999]]
min2 = [[999, 999]]
print("array []:",[vertices])
print("array :",vertices)
#vertices = vertices[:][0][:]
for i in range(4):
    if (max1[0][0] < vertices[i][0]):
        max1[0][0] = vertices[i][0]
        max1[0][1] = vertices[i][1]
    if (min1[0][0] > vertices[i][0]):
        min1[0][0] = vertices[i][0]
        min1[0][1] = vertices[i][1]

for i in range(4):
    if ((max2[0][0] < vertices[i][0]) & (vertices[i][0] < max1[0][0])):
        max2[0][0] = vertices[i][0]
        max2[0][1] = vertices[i][1]
    if ((min2[0][0] > vertices[i][0]) & (vertices[i][0] > min1[0][0])):
        min2[0][0] = vertices[i][0]
        min2[0][1] = vertices[i][1]
print(max1, max2, min1, min2)

if (max1[0][1] > max2[0][1]):
    br = max1
    tr = max2
else:
    br = max2
    tr = max1
if (min1[0][1] > min2[0][1]):
    bl = min1
    tl = min2
else:
    bl = min2
    tl = min1

if display:
    showCorner=[tl,tr,bl,br]
    for i in range(len(showCorner)):
        print(showCorner[i][0])
        m=cv2.circle(img, showCorner[i][0], 2, (255, 0,255), 10)   # vertice independente
        cv2.imshow("Bounding box da caixa e vertices", img)
        cv2.waitKey(0)