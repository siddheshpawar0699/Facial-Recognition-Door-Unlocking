from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget

import cv2
import string
from glob import glob
import pickle

import os.path
import face_recognition
import numpy as np
import sys

from smbus import SMBus

typed_key: string = ""
security_key: string = ""

# I2C Communication
addr = 0x8
# bus = SMBus(1)
lastSentData = -1
slaveDetected = False
# try:
#     # initial test data
#     bus.write_byte(addr, -1)
#     slaveDetected = True
# except IOError:
#     # Slave not found
#     print("Slave Not Found, skipping...")
#     slaveDetected = False

photoLocations = list()

currentPathIndex = 0

# know data
known_face_indexes = list()
known_face_encodings = list()
known_face_paths = list()

process_this_frame = True
save_current_frame = False
set_pass_mode = False
stop_camera = False
know_faces_empty = True

font = cv2.FONT_HERSHEY_DUPLEX

if sys.version_info < (3, 0):
    print("Error: Python2 is slow. Use Python3 for max performance.")
    exit(0)

index_path_data = {}
index_encoding_data = {}


def shutdown():
    global stop_camera
    stop_camera = True
    os.system("sudo shutdown -h now")


def getData():
    global known_face_indexes, known_face_encodings
    global index_path_data, index_encoding_data, known_face_paths
    global know_faces_empty
    global security_key
    know_faces_empty = False
    if os.path.isfile("password.dat"):
        f = open("password.dat", "r")
        security_key = f.readline().strip('\n')
        f.close()
    else:
        security_key = "0000"
        print("Password not found, using default")

    if os.path.isfile("pathData.npy"):
        index_path_data = np.load("pathData.npy", allow_pickle=True).item()
    if os.path.isfile("encodeData.npy"):
        index_encoding_data = np.load("encodeData.npy", allow_pickle=True).item()

    known_face_encodings = list(index_encoding_data.values())
    known_face_indexes = list(index_path_data.keys())
    known_face_paths = list(index_path_data.values())

    if len(index_path_data.keys()) == 0:
        know_faces_empty = True

    print(f'found {len(known_face_encodings)} people faces')


def addFaceToDB():
    global save_current_frame
    save_current_frame = True


def saveFrameToDB(image, encoding):
    global photoLocations, known_face_encodings, known_face_indexes
    global index_path_data, index_encoding_data, known_face_paths
    global know_faces_empty
    pointerIndex = 0
    while os.path.isfile(f"people/{pointerIndex}.jpg"):
        pointerIndex += 1

    cv2.imwrite(f"people/{pointerIndex}.jpg", image)
    print(f'saved! with index {pointerIndex}')
    index_path_data[pointerIndex] = f"people/{pointerIndex}.jpg"
    index_encoding_data[pointerIndex] = encoding
    index_path_data = dict(sorted(index_path_data.items()))
    index_encoding_data = dict(sorted(index_encoding_data.items()))
    known_face_encodings = list(index_encoding_data.values())
    known_face_indexes = list(index_path_data.keys())
    known_face_paths = list(index_path_data.values())
    know_faces_empty = False

    if os.path.isfile("pathData.npy"):
        os.remove("pathData.npy")
    if os.path.isfile("encodeData.npy"):
        os.remove("encodeData.npy")

    np.save("pathData.npy", index_path_data)
    np.save("encodeData.npy", index_encoding_data)

import RPi.GPIO as GPIO
relay_pin=[27]
GPIO.setmode(GPIO.BCM)
GPIO.setup(relay_pin,GPIO.OUT)
GPIO.output(relay_pin,0)

def writeToBUS(index):
    global lastSentData, slaveDetected
    if lastSentData != index:
        lastSentData = index
        relay_pin=[27]
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(relay_pin,GPIO.OUT)
        GPIO.output(relay_pin,1)
        import time
        time.sleep(5)
        GPIO.output(relay_pin,0)
        if (slaveDetected):
            bus.write_byte(addr, index)
    else:
        relay_pin=[27]
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(relay_pin,GPIO.OUT)
        GPIO.output(relay_pin,0)
        pass


getData()


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    callMSG = pyqtSignal(str, str)

    def run(self):
        global stop_camera
        stop_camera = False
        cap = cv2.VideoCapture(0)
        vbool = cap.set(3, 640)
        hbool = cap.set(4, 480)
        print(f'Cam resolution set: {vbool and hbool}')
        if cap is None or not cap.isOpened():
            print('Camera not Available/Supported')
            sys.exit()

        face_locations = []
        global known_face_indexes, known_face_encodings
        global process_this_frame, font, save_current_frame
        global flag
        flag=0
        global lastSentData
        while True:
            from firebase import firebase
            firebase = firebase.FirebaseApplication('https://door-unlock-29cbb-default-rtdb.firebaseio.com/',None)
            result = firebase.get('/value/','')
            if result == 1:
                import time
                print("in")
                relay_pin=[27]
                import RPi.GPIO as GPIO
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(relay_pin,GPIO.OUT)
                GPIO.output(relay_pin,1)
                time.sleep(5)
                GPIO.output(relay_pin,0)
                firebase.put('','value',0)
            
            ret, frame = cap.read()
            if ret:
                if stop_camera:
                    cap.release()
                    return

                face_recognized = False
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                if process_this_frame:
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame)

                    name = "Unknown"
                    face_encodings = []
                    if len(face_locations) == 0:
                        lastSentData = -1

                    if len(face_locations) == 1:  # if face exists
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        for face_encoding in face_encodings:
                            if not know_faces_empty:
                                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,
                                                                         0.5)  # lower = more strict
                                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    flag=1
                                    name = "#" + str(known_face_indexes[best_match_index] + 1)
                                    writeToBUS(known_face_indexes[best_match_index] + 1)
                                    #add Dunction call here to add data base and send email
                                    face_recognized = True
                                else:
                                    flag=0
                   
                    if flag==0:
                        flag=1
                        DIR='unknown'
                        count=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
                        from firebase import firebase
                        firebase = firebase.FirebaseApplication('https://door-unlock-29cbb-default-rtdb.firebaseio.com/',None)
                        firebase.put('','value_unknown',1 )
                        cv2.imwrite("unknown/"+str(count+1)+".png",small_frame)

                    if save_current_frame:
                        if len(face_locations) != 0:
                            if len(face_locations) > 1:
                                self.callMSG.emit("Error", "Multiple Face Found!")
                                save_current_frame = not save_current_frame
                            elif face_recognized:
                                self.callMSG.emit("Error", "Face Already Exists!")
                                save_current_frame = not save_current_frame
                            else:
                                save_current_frame = not save_current_frame
                                self.callMSG.emit("Done", "Face Added!")
                                saveFrameToDB(frame, face_encodings[0])
                        else:
                            self.callMSG.emit("Error", "No Face Found!")
                            save_current_frame = not save_current_frame

                process_this_frame = not process_this_frame

                for top, right, bottom, left in face_locations:
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    if name == "Unknown":
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 255)
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    # Draw a label with a name below the face
                    cv2.putText(frame, name, (left, bottom + 20), font, 0.8, color, 1)

                rgb_og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_og_frame.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgb_og_frame.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(426, 320, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            else:
                print("Error, check if camera is connected!")


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setObjectName("Attendance System")
        self.resize(480, 320)
        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.stackedWidget.setFrameShadow(QtWidgets.QFrame.Plain)
        self.stackedWidget.setObjectName("stackedWidget")
        self.CameraPG = QtWidgets.QWidget()
        self.CameraPG.setObjectName("CameraPG")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.CameraPG)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.videoPRV = QtWidgets.QLabel(self.CameraPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.videoPRV.sizePolicy().hasHeightForWidth())
        self.videoPRV.setSizePolicy(sizePolicy)
        self.videoPRV.setScaledContents(True)
        self.videoPRV.setObjectName("videoPRV")
        self.videoPRV.setAlignment(QtCore.Qt.AlignCenter)
        self.horizontalLayout_4.addWidget(self.videoPRV)

        self.MainButtonGroup = QtWidgets.QVBoxLayout()
        self.MainButtonGroup.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.MainButtonGroup.setSpacing(0)
        self.MainButtonGroup.setObjectName("MainButtonGroup")

        self.settingsBTN = QtWidgets.QPushButton(self.CameraPG)
        icon = QtGui.QPixmap("settings.png")
        self.settingsBTN.setIcon(QtGui.QIcon(icon))
        self.settingsBTN.setIconSize(QtCore.QSize(35, 35))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.settingsBTN.sizePolicy().hasHeightForWidth())
        self.settingsBTN.setSizePolicy(sizePolicy)
        self.settingsBTN.setMaximumSize(QtCore.QSize(54, 16777215))
        self.settingsBTN.setObjectName("settingsBTN")

        self.MainButtonGroup.addWidget(self.settingsBTN)

        self.quitBTN = QtWidgets.QPushButton(self.CameraPG)
        icon2 = QtGui.QPixmap("shutdown.png")
        self.quitBTN.setIcon(QtGui.QIcon(icon2))
        self.quitBTN.setIconSize(QtCore.QSize(30, 30))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.quitBTN.sizePolicy().hasHeightForWidth())
        self.quitBTN.setSizePolicy(sizePolicy)
        self.quitBTN.setMinimumSize(QtCore.QSize(0, 20))
        self.quitBTN.setMaximumSize(QtCore.QSize(54, 16777215))
        self.quitBTN.setObjectName("quitBTN")
        self.MainButtonGroup.addWidget(self.quitBTN)
        self.MainButtonGroup.setStretch(0, 5)
        self.MainButtonGroup.setStretch(1, 1)
        self.horizontalLayout_4.addLayout(self.MainButtonGroup)

        #self.horizontalLayout_4.addWidget(self.settingsBTN)
        self.stackedWidget.addWidget(self.CameraPG)
        self.settingsPG = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.settingsPG.sizePolicy().hasHeightForWidth())
        self.settingsPG.setSizePolicy(sizePolicy)
        self.settingsPG.setObjectName("settingsPG")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.settingsPG)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.backBTN = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.backBTN.sizePolicy().hasHeightForWidth())
        self.backBTN.setSizePolicy(sizePolicy)
        self.backBTN.setObjectName("backBTN")
        self.horizontalLayout_3.addWidget(self.backBTN)
        self.VL1 = QtWidgets.QVBoxLayout()
        self.VL1.setSpacing(5)
        self.VL1.setObjectName("VL1")
        self.passwordTTL = QtWidgets.QLabel(self.settingsPG)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.passwordTTL.setFont(font)
        self.passwordTTL.setFrameShadow(QtWidgets.QFrame.Plain)
        self.passwordTTL.setTextFormat(QtCore.Qt.PlainText)
        self.passwordTTL.setAlignment(QtCore.Qt.AlignCenter)
        self.passwordTTL.setObjectName("passwordTTL")
        self.VL1.addWidget(self.passwordTTL)
        self.passwordENT = QtWidgets.QLineEdit(self.settingsPG)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.passwordENT.setFont(font)
        self.passwordENT.setAlignment(QtCore.Qt.AlignCenter)
        self.passwordENT.setReadOnly(False)
        self.passwordENT.setObjectName("passwordENT")
        self.VL1.addWidget(self.passwordENT)
        self.numpad = QtWidgets.QGridLayout()
        self.numpad.setSpacing(4)
        self.numpad.setObjectName("numpad")
        self.btn_5 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_5.sizePolicy().hasHeightForWidth())
        self.btn_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_5.setFont(font)
        self.btn_5.setObjectName("btn_5")
        self.numpad.addWidget(self.btn_5, 1, 1, 1, 1)
        self.btn_2 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_2.sizePolicy().hasHeightForWidth())
        self.btn_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_2.setFont(font)
        self.btn_2.setObjectName("btn_2")
        self.numpad.addWidget(self.btn_2, 0, 1, 1, 1)
        self.btn_3 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_3.sizePolicy().hasHeightForWidth())
        self.btn_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_3.setFont(font)
        self.btn_3.setObjectName("btn_3")
        self.numpad.addWidget(self.btn_3, 0, 2, 1, 1)
        self.btn_4 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_4.sizePolicy().hasHeightForWidth())
        self.btn_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_4.setFont(font)
        self.btn_4.setObjectName("btn_4")
        self.numpad.addWidget(self.btn_4, 1, 0, 1, 1)
        self.btn_1 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_1.sizePolicy().hasHeightForWidth())
        self.btn_1.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_1.setFont(font)
        self.btn_1.setObjectName("btn_1")
        self.numpad.addWidget(self.btn_1, 0, 0, 1, 1)
        self.btn_6 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_6.sizePolicy().hasHeightForWidth())
        self.btn_6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_6.setFont(font)
        self.btn_6.setObjectName("btn_6")
        self.numpad.addWidget(self.btn_6, 1, 2, 1, 1)
        self.btn_8 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_8.sizePolicy().hasHeightForWidth())
        self.btn_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_8.setFont(font)
        self.btn_8.setObjectName("btn_8")
        self.numpad.addWidget(self.btn_8, 2, 1, 1, 1)
        self.btn_9 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_9.sizePolicy().hasHeightForWidth())
        self.btn_9.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_9.setFont(font)
        self.btn_9.setObjectName("btn_9")
        self.numpad.addWidget(self.btn_9, 2, 2, 1, 1)
        self.btn_7 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_7.sizePolicy().hasHeightForWidth())
        self.btn_7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_7.setFont(font)
        self.btn_7.setObjectName("btn_7")
        self.numpad.addWidget(self.btn_7, 2, 0, 1, 1)
        self.btn_0 = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_0.sizePolicy().hasHeightForWidth())
        self.btn_0.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_0.setFont(font)
        self.btn_0.setObjectName("btn_0")
        self.numpad.addWidget(self.btn_0, 3, 1, 1, 1)
        self.btn_clear_all = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_clear_all.sizePolicy().hasHeightForWidth())
        self.btn_clear_all.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_clear_all.setFont(font)
        self.btn_clear_all.setObjectName("btn_clear")
        self.numpad.addWidget(self.btn_clear_all, 3, 0, 1, 1)
        self.btn_clear = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_clear.sizePolicy().hasHeightForWidth())
        self.btn_clear.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_clear.setFont(font)
        self.btn_clear.setObjectName("btn_sub")
        self.numpad.addWidget(self.btn_clear, 3, 2, 1, 1)
        self.VL1.addLayout(self.numpad)
        self.VL1.setStretch(0, 10)
        self.VL1.setStretch(1, 10)
        self.VL1.setStretch(2, 80)
        self.horizontalLayout_3.addLayout(self.VL1)
        self.doneBTN = QtWidgets.QPushButton(self.settingsPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.doneBTN.sizePolicy().hasHeightForWidth())
        self.doneBTN.setSizePolicy(sizePolicy)
        self.doneBTN.setObjectName("doneBTN")
        self.horizontalLayout_3.addWidget(self.doneBTN)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 98)
        self.horizontalLayout_3.setStretch(2, 1)
        self.stackedWidget.addWidget(self.settingsPG)
        self.ChangesPG = QtWidgets.QWidget()
        self.ChangesPG.setObjectName("ChangesPG")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.ChangesPG)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.setpassBTN = QtWidgets.QPushButton(self.ChangesPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.setpassBTN.sizePolicy().hasHeightForWidth())
        self.setpassBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.setpassBTN.setFont(font)
        self.setpassBTN.setObjectName("setpassBTN")
        self.gridLayout_2.addWidget(self.setpassBTN, 1, 1, 1, 1)
        self.homeBTN = QtWidgets.QPushButton(self.ChangesPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.homeBTN.sizePolicy().hasHeightForWidth())
        self.homeBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.homeBTN.setFont(font)
        self.homeBTN.setObjectName("homeBTN")
        self.gridLayout_2.addWidget(self.homeBTN, 1, 0, 1, 1)
        self.addfaceBTN = QtWidgets.QPushButton(self.ChangesPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.addfaceBTN.sizePolicy().hasHeightForWidth())
        self.addfaceBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.addfaceBTN.setFont(font)
        self.addfaceBTN.setObjectName("addfaceBTN")
        self.gridLayout_2.addWidget(self.addfaceBTN, 2, 0, 1, 1)
        self.removefaceBTN = QtWidgets.QPushButton(self.ChangesPG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.removefaceBTN.sizePolicy().hasHeightForWidth())
        self.removefaceBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.removefaceBTN.setFont(font)
        self.removefaceBTN.setObjectName("removefaceBTN")
        self.gridLayout_2.addWidget(self.removefaceBTN, 2, 1, 1, 1)
        self.stackedWidget.addWidget(self.ChangesPG)
        self.AddFace = QtWidgets.QWidget()
        self.AddFace.setObjectName("AddFace")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.AddFace)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.videoPRVADD = QtWidgets.QLabel(self.AddFace)
        self.videoPRVADD.setMinimumSize(QtCore.QSize(426, 0))
        self.videoPRVADD.setAlignment(QtCore.Qt.AlignCenter)
        self.videoPRVADD.setScaledContents(True)
        self.videoPRVADD.setObjectName("cameraPRVA")
        self.horizontalLayout_5.addWidget(self.videoPRVADD)
        self.VL3 = QtWidgets.QVBoxLayout()
        self.VL3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.VL3.setSpacing(0)
        self.VL3.setObjectName("VL3")
        self.add_backBTN = QtWidgets.QPushButton(self.AddFace)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.add_backBTN.sizePolicy().hasHeightForWidth())
        self.add_backBTN.setSizePolicy(sizePolicy)
        self.add_backBTN.setMaximumSize(QtCore.QSize(61, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.add_backBTN.setFont(font)
        self.add_backBTN.setObjectName("add_backBTN")
        self.VL3.addWidget(self.add_backBTN)
        self.savefaceBTN = QtWidgets.QPushButton(self.AddFace)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.savefaceBTN.sizePolicy().hasHeightForWidth())
        self.savefaceBTN.setSizePolicy(sizePolicy)
        self.savefaceBTN.setMaximumSize(QtCore.QSize(61, 16777215))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.savefaceBTN.setFont(font)
        self.savefaceBTN.setObjectName("savefaceBTN")
        self.VL3.addWidget(self.savefaceBTN)
        self.horizontalLayout_5.addLayout(self.VL3)
        self.stackedWidget.addWidget(self.AddFace)

        # added pg3
        self.removeFacePG = QtWidgets.QWidget()
        self.removeFacePG.setObjectName("removeFacePG")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.removeFacePG)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.HashNumber = QtWidgets.QLabel(self.removeFacePG)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.HashNumber.setFont(font)
        self.HashNumber.setAlignment(QtCore.Qt.AlignCenter)
        self.HashNumber.setObjectName("HashNumber")
        self.verticalLayout_2.addWidget(self.HashNumber)
        self.ImagePV = QtWidgets.QLabel(self.removeFacePG)
        self.ImagePV.setAlignment(QtCore.Qt.AlignCenter)
        self.ImagePV.setObjectName("ImagePV")
        self.verticalLayout_2.addWidget(self.ImagePV)
        self.HL1 = QtWidgets.QHBoxLayout()
        self.HL1.setObjectName("HL1")
        self.remove_backBTN = QtWidgets.QPushButton(self.removeFacePG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.remove_backBTN.sizePolicy().hasHeightForWidth())
        self.remove_backBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.remove_backBTN.setFont(font)
        self.remove_backBTN.setObjectName("remove_backBTN")
        self.HL1.addWidget(self.remove_backBTN)
        self.previousBTN = QtWidgets.QPushButton(self.removeFacePG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.previousBTN.sizePolicy().hasHeightForWidth())
        self.previousBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.previousBTN.setFont(font)
        self.previousBTN.setObjectName("previousBTN")
        self.HL1.addWidget(self.previousBTN)
        self.nextBTN = QtWidgets.QPushButton(self.removeFacePG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nextBTN.sizePolicy().hasHeightForWidth())
        self.nextBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.nextBTN.setFont(font)
        self.nextBTN.setObjectName("nextBTN")
        self.HL1.addWidget(self.nextBTN)
        self.deleteBTN = QtWidgets.QPushButton(self.removeFacePG)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.deleteBTN.sizePolicy().hasHeightForWidth())
        self.deleteBTN.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.deleteBTN.setFont(font)
        self.deleteBTN.setObjectName("deleteBTN")
        self.HL1.addWidget(self.deleteBTN)
        self.verticalLayout_2.addLayout(self.HL1)
        self.verticalLayout_2.setStretch(0, 8)
        self.verticalLayout_2.setStretch(1, 2)
        self.stackedWidget.addWidget(self.removeFacePG)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.stackedWidget.addWidget(self.page_2)

        self.horizontalLayout.addWidget(self.stackedWidget)
        self.setCentralWidget(self.centralwidget)

        self.retranslateUi()
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.videoPRV.setText(_translate("MainWindow", "camera"))
        self.backBTN.setText(_translate("MainWindow", "Back"))
        self.passwordTTL.setText(_translate("MainWindow", "Security Key"))
        self.btn_5.setText(_translate("MainWindow", "5"))
        self.btn_2.setText(_translate("MainWindow", "2"))
        self.btn_3.setText(_translate("MainWindow", "3"))
        self.btn_4.setText(_translate("MainWindow", "4"))
        self.btn_1.setText(_translate("MainWindow", "1"))
        self.btn_6.setText(_translate("MainWindow", "6"))
        self.btn_8.setText(_translate("MainWindow", "8"))
        self.btn_9.setText(_translate("MainWindow", "9"))
        self.btn_7.setText(_translate("MainWindow", "7"))
        self.btn_0.setText(_translate("MainWindow", "0"))

        self.btn_0.clicked.connect(lambda: self.append_key("0"))
        self.btn_1.clicked.connect(lambda: self.append_key("1"))
        self.btn_2.clicked.connect(lambda: self.append_key("2"))
        self.btn_3.clicked.connect(lambda: self.append_key("3"))
        self.btn_4.clicked.connect(lambda: self.append_key("4"))
        self.btn_5.clicked.connect(lambda: self.append_key("5"))
        self.btn_6.clicked.connect(lambda: self.append_key("6"))
        self.btn_7.clicked.connect(lambda: self.append_key("7"))
        self.btn_8.clicked.connect(lambda: self.append_key("8"))
        self.btn_9.clicked.connect(lambda: self.append_key("9"))

        self.homeBTN.clicked.connect(lambda: self.setStackPage(0))
        self.setpassBTN.clicked.connect(lambda: self.set_pass_mode())
        self.settingsBTN.clicked.connect(lambda: self.setStackPage(1))
        self.quitBTN.clicked.connect(lambda: shutdown())
        self.backBTN.clicked.connect(lambda: self.setStackPage(0))
        self.btn_clear.clicked.connect(lambda: self.clear_key())
        self.btn_clear_all.clicked.connect(lambda: self.clear_all_key())
        self.doneBTN.clicked.connect(lambda: self.check_set_key())
        self.passwordENT.setEchoMode(QtWidgets.QLineEdit.Password)
        self.addfaceBTN.clicked.connect(lambda: self.setStackPage(3))
        self.add_backBTN.clicked.connect(lambda: self.setStackPage(2))
        self.removefaceBTN.clicked.connect(lambda: self.setStackPage(4))
        self.remove_backBTN.clicked.connect(lambda: self.setStackPage(2))
        self.nextBTN.clicked.connect(lambda: self.loadNextPhoto())
        self.previousBTN.clicked.connect(lambda: self.loadPreviousPhoto())
        self.savefaceBTN.clicked.connect(lambda: addFaceToDB())
        self.deleteBTN.clicked.connect(lambda: self.deleteFacefromDB())

        self.btn_clear_all.setText(_translate("MainWindow", "clear"))
        self.btn_clear.setText(_translate("MainWindow", "<"))
        self.doneBTN.setText(_translate("MainWindow", "Done"))
        self.setpassBTN.setText(_translate("MainWindow", "Set Security Key"))
        self.homeBTN.setText(_translate("MainWindow", "Home"))
        self.addfaceBTN.setText(_translate("MainWindow", "Add Face"))
        self.removefaceBTN.setText(_translate("MainWindow", "Remove Face"))
        self.videoPRVADD.setText(_translate("MainWindow", "camera"))
        self.add_backBTN.setText(_translate("MainWindow", "Back"))
        self.savefaceBTN.setText(_translate("MainWindow", "+"))

        # added pg3
        self.HashNumber.setText(_translate("MainWindow", "#0"))
        self.ImagePV.setText(_translate("MainWindow", "Image"))
        self.remove_backBTN.setText(_translate("MainWindow", "Back"))
        self.previousBTN.setText(_translate("MainWindow", "<"))
        self.nextBTN.setText(_translate("MainWindow", ">"))
        self.deleteBTN.setText(_translate("MainWindow", "Delete"))

        self.startCameraThread()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.videoPRV.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def setImageADD(self, image):
        self.videoPRVADD.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(str, str)
    def showCannotAddMSG(self, note, message):
        QMessageBox.about(self, note, message)

    def startCameraThread(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    def startAddFaceCameraThread(self):
        th = Thread(self)
        th.changePixmap.connect(self.setImageADD)
        th.callMSG.connect(self.showCannotAddMSG)
        th.start()

    def setStackPage(self, num_pg: int):
        global typed_key, set_pass_mode, stop_camera
        if num_pg == 0 or num_pg == 2:
            self.passwordENT.setText("")
            if set_pass_mode:
                set_pass_mode = False
                self.passwordTTL.setText("Security Key")
        if num_pg == 0:
            self.startCameraThread()
        if num_pg == 3:
            self.startAddFaceCameraThread()
        if num_pg == 4:
            global known_face_paths
            if len(known_face_paths) == 0:
                QMessageBox.about(self, "Error", "No Face Found!")
                return
            else:
                self.loadPhoto()

        stop_camera = True
        typed_key = ""
        self.stackedWidget.setCurrentIndex(num_pg)

    def set_pass_mode(self):
        global set_pass_mode
        set_pass_mode = True
        self.passwordTTL.setText("Set Security Key")
        self.setStackPage(1)

    def check_set_key(self):
        global typed_key, security_key
        if set_pass_mode:
            if len(typed_key) == 4:
                security_key = typed_key
                f = open("password.dat", "w")
                print(security_key, file=f)
                f.close()
                self.setStackPage(2)
            else:
                QMessageBox.about(self, "Error", "Minimum 4 keys!")
        else:
            if typed_key == security_key:
                self.setStackPage(2)
            else:
                QMessageBox.about(self, "Error", "Wrong Password!")
                self.clear_all_key()

    def clear_all_key(self):
        global typed_key
        typed_key = ""
        self.passwordENT.setText(typed_key)

    def append_key(self, ka: string):
        global typed_key
        if len(typed_key) >= 4:
            return
        typed_key += ka
        self.passwordENT.setText(typed_key)

    def clear_key(self):
        global typed_key
        typed_key = typed_key[0:-1]
        self.passwordENT.setText(typed_key)

    def loadPhoto(self):
        global currentPathIndex, known_face_paths
        self.handlePhoto()

    def loadNextPhoto(self):
        global currentPathIndex, known_face_paths
        if currentPathIndex != len(known_face_paths) - 1:
            currentPathIndex += 1
        self.handlePhoto()

    def loadPreviousPhoto(self):
        global currentPathIndex, known_face_paths
        if currentPathIndex != 0:
            currentPathIndex -= 1
        self.handlePhoto()

    def handlePhoto(self):
        global currentPathIndex, known_face_paths, known_face_indexes
        if len(known_face_paths) == 0:
            self.setStackPage(2)
            QMessageBox.about(self, "Error", "No Face Found!")
            return
        pixmap = QPixmap(known_face_paths[currentPathIndex])
        pixmap = pixmap.scaled(480, 256, QtCore.Qt.KeepAspectRatio)
        self.ImagePV.setPixmap(pixmap)
        self.HashNumber.setText("#" + str(known_face_indexes[currentPathIndex] + 1))

    def deleteFacefromDB(self):
        global currentPathIndex, photoLocations, known_face_encodings, known_face_indexes
        global index_path_data, index_encoding_data, known_face_paths
        global know_faces_empty
        os.remove(known_face_paths[currentPathIndex])
        index_path_data.pop(known_face_indexes[currentPathIndex])
        index_encoding_data.pop(known_face_indexes[currentPathIndex])
        known_face_encodings = list(index_encoding_data.values())
        known_face_indexes = list(index_path_data.keys())
        known_face_paths = list(index_path_data.values())
        if len(known_face_paths) == 0:
            know_faces_empty = True

        if os.path.isfile("pathData.npy"):
            os.remove("pathData.npy")
        if os.path.isfile("encodeData.npy"):
            os.remove("encodeData.npy")

        np.save("pathData.npy", index_path_data)
        np.save("encodeData.npy", index_encoding_data)
        QMessageBox.about(self, "Done", "Face Deleted!")
        self.loadPreviousPhoto()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
