from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from Modelo.predict import make_prediction
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import fins.udp
import os
import sys
import cv2
import pandas as pd
import numpy as np
from speech_recognition import Microphone, Recognizer, AudioFile, UnknownValueError, RequestError
from tensorflow.keras.models import load_model
from threading import Thread, Event
import face_recognition
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from meds_ocr_output_func import ocrOut
import warnings
import time
warnings.filterwarnings("ignore")

fins_instance = fins.udp.UDPFinsConnection()

fins_instance.dest_node_add=101 # IP do plc ip 192.168.1.101
fins_instance.srce_node_add=203 # IP do host-- pc ip do pc 192.168.1.203

excelDiretorio = 'C:\\Users\\Raul\\PycharmProjects\\PyCharm PyQt5\\allPackages.xls'
worksheetName = 'Medicamentos'
MedPath="allPackages1.csv"
modelface = load_model('models/MobileNet_Caras_60epoch.h5')

cap2 = cv2.VideoCapture(0)

def img2pixmap(image):  # Camara na Interface
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    qimage = QImage(image.data, width, height, bytesPerLine, QImage.Format_BGR888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap



def grabFrame():  # Camara na Interface
    if not cap2.isOpened():
        cap2.open(0)
    ret, image = cap2.read()
    window.faceCamera.setPixmap(img2pixmap(image))

def camara():
    cap2 = cv2.VideoCapture(1)

    for i in range(300):
        ret, img = cap2.read()
        window.medsCameraLabel.setPixmap(img2pixmap(img))
        cv2.imwrite("Meds/medicamento.jpg", img)
        cv2.waitKey(1)
    cap2.release()


def camaraFace():
    cap2 = cv2.VideoCapture(0)
    for i in range(100):
        ret, img = cap2.read()
        window.faceCamera.setPixmap(img2pixmap(img))
        cv2.imwrite("FaceDir1/face.jpg", img)
        cv2.waitKey(1)
    cap2.release()

def on_cameraON_clicked():  # Camara na Interface
    qtimerFrame.start(50)

def detectKeyword():
    recognizer = Recognizer()
    Login = 0
    Classe = 0
    pagina = 0
    limpar=0
    df = pd.read_csv(MedPath, sep='delimiter')
    x = df['Nome do medicamento'].unique()
    while True:

        try:
            with Microphone() as mic:
                print("Talk")
                recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                # audio = recognizer.listen(mic)

                audio = recognizer.listen(mic, None, 5)
                text = recognizer.recognize_google(audio, None, "pt-PT")
                text = text.lower()
                print(f"Recognized {text}")
                Login=1
                # ("alfredo" in text) &
                if (("alfredo" == text) & (Login == 0)):
                    # if(Login == 0):
                    with open('C:\\Users\\Raul\\PycharmProjects\\PyCharm PyQt5\\Keyword/speech.wav', 'wb') as f:
                        f.write(audio.get_wav_data())
                        os.system(f"ffmpeg -i   \"Keyword/speech.wav\" -ac 1 -ar 16000 \"Audios/Unknown/speech.wav\"")
                        window.stackedWidget.setCurrentWidget(window.voiceRecPage)
                        Classe = make_prediction()
                        info(Classe)
                        os.remove("Audios/Unknown/speech.wav")
                        window.stackedWidget.setCurrentWidget(window.faceRecPage)
                        Login = detectFace(Classe)
                        if Login == 0:
                            window.stackedWidget.setCurrentWidget(window.loginPage)
                        else:
                            window.stackedWidget.setCurrentWidget(window.homePage)
                            pagina = 1
                elif("home" == text):
                    if(Login == 1):
                        window.stackedWidget.setCurrentWidget(window.homePage)
                        window.label_22.setText(f"<html><head/><body><p align=\"center\"><span style=\" font-size:48pt; color:#ebebeb;\">Order your med</span></p></body></html>")
                        pagina = 1
                elif ("info" == text):
                    if (Login == 1):
                        window.stackedWidget.setCurrentWidget(window.InfoPage)
                        pagina = 2
                elif ("medicamentos" == text):
                    if (Login == 1):
                        window.stackedWidget.setCurrentWidget(window.medsPage)
                        pagina = 3
                elif ("armazém" == text):
                    if (Login == 1):
                        window.stackedWidget.setCurrentWidget(window.orderPage)
                        tabelaMeds(excelDiretorio,worksheetName)
                        pagina = 4
                elif(pagina == 1):
                    for i in range(x.size):
                        if (x[i].lower() in text):
                            window.label_22.setText(f"<html><head/><body><p align=\"center\"><span style=\" font-size:48pt; color:#ebebeb;\">{x[i]}</span></p></body></html>")
                            break
                elif(pagina == 3):
                    if ("iniciar" in text):
                        camara()
                        img = cv2.imread("Meds/medicamento.jpg")
                        med_text = ocrOut(img)
                        for i in range(x.size):
                            if(x[i].lower() in med_text):
                                window.medsName.setText(f"<html><head/><body><p align=\"center\"><span style=\" font-size:48pt; color:#ebebeb;\">{x[i]}</span></p></body></html>")
                                break
                            fins_instance.Init(50)
                elif(pagina == 4):
                    for i in range(x.size):
                        if(text in x[i].lower()):
                            if limpar == 1:
                                window.tableWidget.setRowCount(0)
                                tabelaMeds(excelDiretorio, worksheetName)
                                limpar = 0
                    if(limpar == 0):
                        for i in range(x.size):
                            if(text in x[i].lower()):
                                findName(text)
                                limpar=1
        except:
            print("Error, no microphone detected")

def detectFace(Classe):
    camaraFace()
    crop_image()
    y_pred=predict_face()
    print(y_pred)
    if(y_pred==Classe):
        Login =1
    else:
        Login=0
    return Login

def crop_image():
    img = cv2.imread("FaceDir1/face.jpg")
    imgCrop = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    face_locations = face_recognition.face_locations(imgCrop)

    for indexf, faces in enumerate(
            face_locations):  # permite fazer como um if() pq só vai fazer crop se houver caras  (será necessario?)
        # crop
        y, x2, y2, x = faces  # face_locations  #top, right, bottom, left
        w = x2 - x
        h = y2 - y
        imgCrop = imgCrop[y:y + h, x:x + w]
    cv2.imwrite("Face_Dir/Unknown/face1.jpg", imgCrop)

def predict_face():
    val_dir = "C:\\Users\\Raul\\PycharmProjects\\PyCharm PyQt5\\Face_Dir"
    image_gen_test = ImageDataGenerator(rescale=1. / 255)
    test_data_gen = image_gen_test.flow_from_directory(batch_size=16, directory=val_dir, target_size=(224, 224), class_mode='sparse')
    X_test, y_test = test_data_gen.next()
    Y_predict = modelface.predict(X_test)
    y_pred = np.argmax(Y_predict)
    # show
    print(y_pred)
    return y_pred

def threadVoice():
    Thread(target=detectKeyword).start()

def tabelaMeds(excelDir, worksheet):
    df = pd.read_excel(excelDir, worksheet)
    if(df.size == 0):
        return


    window.tableWidget.setRowCount(df.shape[0])
    window.tableWidget.setColumnCount(df.shape[1])
    window.tableWidget.setHorizontalHeaderLabels(df.columns)

    for row in df.iterrows():
        values = row[1]
        for col_index, value in enumerate(values):
            if isinstance(value, (float,int)):
                value = '{0.0,.0f}'.format(value)
            tableItem = QTableWidgetItem(str(value))
            window.tableWidget.setItem(row[0], col_index, tableItem)

    window.tableWidget.resizeColumnsToContents()
    window.tableWidget.resizeRowsToContents()
    window.tableWidget.verticalHeader().setVisible(False)

def findName(name):
    for row in range(window.tableWidget.rowCount()):
        if name not in window.tableWidget.item(row, 0).text().lower():
            window.tableWidget.hideRow(row)



app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("loginWindow.ui")

# PLC
fins_instance.Init(10)
time.sleep(5)
fins_instance.Init(99)

mem_area=fins_instance.Read_Sate(20)
if(mem_area == 'aaaa'):
    print("Connect")
else:
    print("Error Connecting")



threadVoice()




#Info ativar camara
window.recognizeBtn.clicked.connect(camara)
window.medsCameraLabel.setScaledContents(True)
window.faceCamera.setScaledContents(False)

#Home para Order
window.orderBtn.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.orderPage))
window.orderBtn.clicked.connect(lambda: tabelaMeds(excelDiretorio,worksheetName))

#Info para Order
window.orderBtn_2.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.orderPage))
window.orderBtn_2.clicked.connect(lambda: tabelaMeds(excelDiretorio,worksheetName))

#Meds para Order
window.orderBtn_3.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.orderPage))
window.orderBtn_3.clicked.connect(lambda: tabelaMeds(excelDiretorio,worksheetName))

if window.nomeMed.text() != "":
    findName()

#Show Med request

def info(Classe):
    if Classe == 0:
        # IF LOGIN JOAQUIM Informações
        window.welcomeLabel.setText("<html><head/><body><p align=\"center\"><span style=\" font-size:48pt; color:#ebebeb;\">Joaquim<br/></span></p></body></html>")
        pixmap3 = QPixmap("joaquim.png")
        window.photoLabel.setPixmap(pixmap3)
        joaquimInfo = QPixmap("joaquimInfo.png")
        window.infoLabel.setPixmap(joaquimInfo)
    elif Classe == 1:
        #IF LOGIN PONTES Informações
        window.welcomeLabel.setText("<html><head/><body><p align=\"center\"><span style=\" font-size:48pt; color:#ebebeb;\">Pontes<br/></span></p></body></html>")
        pixmap = QPixmap("pontes.png")
        window.photoLabel.setPixmap(pixmap)
        pontesInfo = QPixmap("pontesInfo.png")
        window.infoLabel.setPixmap(pontesInfo)
    elif Classe == 2:
        #IF LOGIN RAUL Informações
        window.welcomeLabel.setText("<html><head/><body><p align=\"center\"><span style=\" font-size:48pt; color:#ebebeb;\">Raúl<br/></span></p></body></html>")
        pixmap1 = QPixmap("raul.png")
        window.photoLabel.setPixmap(pixmap1)
        raulInfo = QPixmap("raulInfo.png")
        window.infoLabel.setPixmap(raulInfo)
    elif Classe == 3:
        #IF LOGIN VALE Informações
        window.welcomeLabel.setText("<html><head/><body><p align=\"center\"><span style=\" font-size:48pt; color:#ebebeb;\">Diogo<br/></span></p></body></html>")
        pixmap2 = QPixmap("vale.png")
        window.photoLabel.setPixmap(pixmap2)
        valeInfo = QPixmap("valeInfo.png")
        window.infoLabel.setPixmap(valeInfo)




#Mudar para janela Home
window.homeBtn_2.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.homePage))
window.homeBtn_3.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.homePage))
window.homeBtn_4.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.homePage))

#Mudar para janela de Info
window.infoBtn.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.InfoPage))
window.infoBtn_3.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.InfoPage))
window.infoBtn_4.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.InfoPage))

#Mudar para janela Meds
window.medsBtn.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.medsPage))
window.medsBtn_2.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.medsPage))
window.medsBtn_4.clicked.connect(lambda: window.stackedWidget.setCurrentWidget(window.medsPage))

qtimerFrame = QTimer()
qtimerFrame.timeout.connect(grabFrame)
window.show()
app.exec()