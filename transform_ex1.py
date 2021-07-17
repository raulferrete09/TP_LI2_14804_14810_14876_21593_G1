
# WORKS to unwarp or transform a rectangular object in an image if given the vertices !!!!


aspectRatio=True
testing=True
save=False

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math as m

def unwarp(img, src, dst, newSize, testing):
    h, w = img.shape[:2]
    #newSize = (w, h)
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, newSize, flags=cv2.INTER_LINEAR)

    if testing:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        f.subplots_adjust(hspace=.2, wspace=.05)
        ax1.imshow(img, 'gray')
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(warped, 'gray')
        ax2.set_title('Unwarped Image', fontsize=30)
        plt.show()
        # Saving an image
        if(save):
            cv2.imwrite("/Users/diogo/Desktop/Cadeiras/Mestrado/labs II/scripts/Imagem/Meds/Images/benuron_from_net_unwarped2.jpg", warped)
            print("unwarped SAVED")
    else:
        return warped, M

im = cv2.imread("/Users/diogo/Desktop/Cadeiras/Mestrado/labs II/scripts/Imagem/Meds/Images/benuron_from_net_40perCent.jpg", cv2.IMREAD_GRAYSCALE)
h, w = im.shape[:2]

# object vertices coordinates [x,y]
tl=[40, 155]
tr=[575, 55]
bl=[105, 460]
br=[630, 355]

##  CODIGO para obter uma transformação mantendo o aspectRatio do objeto rectangular
if aspectRatio:
    print("Aspect Ratio Mode: ON !!!!")
    # get object height and width
    leftSize = int(m.sqrt( (bl[0]-tl[0])**2 + (bl[1]-tl[1])**2 )); #print(leftSize)
    rightSize = int(m.sqrt( (br[0]-tr[0])**2 + (br[1]-tr[1])**2 )); #print(rightSize)
    topSize = int(m.sqrt( (tl[0]-tr[0])**2 + (tl[1]-tr[1])**2 )); #print(topSize)
    bottomSize = int(m.sqrt( (bl[0]-br[0])**2 + (bl[1]-br[1])**2 )); #print(bottomSize)
    # salva a maior largura e comprimento
    h = leftSize if leftSize>rightSize else rightSize
    w = topSize if topSize>bottomSize else bottomSize


src = np.float32([(bl[0], bl[1]),     #bottom left
                  (tl[0], tl[1]),     #top left
                  (br[0], br[1]),     #bottom right
                  (tr[0], tr[1])])    #top right

print("output height: ", h, "\twidth", w)
dst = np.float32([(0, h),(0, 0),(w, h),(w, 0)])

[img, m]=unwarp(im, src, dst, (w,h), testing)

# delete in implementation !!!!!
if not testing:
    cv2.imshow("cropped", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
