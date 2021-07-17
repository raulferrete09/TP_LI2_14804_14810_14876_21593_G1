'''import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)
'''

# load Images
from os import listdir
from PIL import Image as PImage
# IP
import cv2
import numpy as np
import face_recognition
import os


# load Images
def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        a = PImage.open(path + image)
        img = np.asarray(a)
        loadedImages.append(img)

    return loadedImages


path = "C:/Users/diogo/Desktop/Cadeiras/Mestrado/labs II/scripts/Imagem/Face/FaceClassifier/Images/"
savePath = "C:/Users/diogo/Desktop/Cadeiras/Mestrado/labs II/scripts/Imagem/Face/FaceClassifier/Faces/"
classes = {"Vale"}
# {"Joaquim", "Mask_Pontes", "Mask_Raul", "Mask_Vale"}
nImg = 0; nFaces = 0
ntImg = 0; ntFaces = 0

for folder in classes:

    # images Array
    imgs = loadImages(path + folder + "/")
    nImg = len(imgs)  # image counter
    nFaces=0
    print(folder)
    for index, img in enumerate(imgs):
        # detect faces
        # IP from the used code

        #print(img.shape, end="")
        if(img.shape[0]==1932):
            scale_percent = 60  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            #img = img.resize(img,(width, height))

            img = cv2.resize(img, (width, height))
            #print("\t",img.shape)
        #else:
            #print("\t"+str(index)+"\t",end="")
        imgCrop = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(imgCrop)  # QUESTÕES MAIS A BAIXO  !!!!!!!!!

        for indexf,faces in enumerate(face_locations):  # permite fazer como um if() pq só vai fazer crop se houver caras  (será necessario?)
            # crop
            y, x2, y2, x = faces  # face_locations  #top, right, bottom, left
            w = x2 - x
            h = y2 - y
            imgCrop = imgCrop[y:y + h, x:x + w]
            #cv2.imshow("image", img)
            #cv2.imshow("Cropped face", imgCrop)

            # save
            #print(len(face_locations),indexf)
            if img.shape[0]*img.shape[1]!=0:
                if indexf==0:
                    imgCropName = folder + "_" + str(index) + ".jpg"
                    cv2.imwrite(savePath + folder + "/" + imgCropName, imgCrop)
                #else:
                 #   imgCropName = folder + "_" + str(index) + "(" + str(indexf) + ")" + ".jpg"
                  #  cv2.imwrite(savePath + folder + "/" + imgCropName, imgCrop)

            nFaces += 1


    if(nFaces!=0):
        print(folder,"faces found: ", nFaces / nImg * 100, "%  (",nImg,")")
        ntFaces+=nFaces
        ntImg+=nImg
print("Total faces found: ", ntFaces / ntImg * 100, "%  (",ntImg,")")


'''
1) tanto se pode criar um modelo de de classificação das imagens (crop da cara, o q seria "raw data") ou 
treinar o modelp para classificar
2) o que acontece quando há mais q uma cara na imagem???
(se houver mais q uma o face_locations() vai retornar mais q uma bounding box)
 -ver como o codigo original funciona mas mais q uma cara!!!
'''

# IP
path = 'imageBasic'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImage = cv2.imread(f'{path}/{cl}')
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncondings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encondeListKnown = findEncondings(images)
print(len(encondeListKnown))

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encondeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encondeListKnown, encodeFace)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
