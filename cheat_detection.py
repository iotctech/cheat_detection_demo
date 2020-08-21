from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
import cv2
import time
from log import LogThread
from image_handle import ResnetDecode
import paddlehub as hub

# 采用线程来播放视频
class dpThread(QThread):
    changePixmap = pyqtSignal(QImage)
    eventPush = pyqtSignal(str)

    def __init__(self, parent=None, logThread=None, resnetModel=None, faceDetectModel=None):
        super(dpThread, self).__init__(parent)
        self.logThread = logThread
        self.needStop = False
        self.resnetModel = resnetModel
        self.faceDetectModel = faceDetectModel

    def run(self):
        camera = cv2.VideoCapture(0)
        index = 0
        while (camera.isOpened()==True):
            res, frame = camera.read()
            if res:
                frame = cv2.resize(frame, (700, 500))
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                p = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], rgbImage.shape[1]*3, QImage.Format_RGB888)
                self.changePixmap.emit(p)

                # out = self.resnetModel.predict(frame, frame.shape)
                # if out[0][0][0] > out[0][0][1]:
                #     print("cheating")
                # else:
                #     print("uncheating")

                time.sleep(0.01) #控制视频播放的速度
                index += 1
                if index > 250:
                    index = 0
                    # self.logThread.printf("[系统消息]：作弊检测中...")
                    result = self.faceDetectModel.face_detection(images=[frame])
                    if result[0]['data'] != []:
                        self.logThread.printf("[系统消息]：检测到人脸")
                    else:
                        self.logThread.printf("[预警消息]：未检测到人脸")
                        self.eventPush.emit("空位事件")

                    out = self.resnetModel.predict(frame, frame.shape)
                    if out[0][0][0] > out[0][0][1]:
                        # print("cheating")
                        self.logThread.printf("[预警消息]：头部姿态异常")
                        self.eventPush.emit("作弊事件")
                    else:
                        # print("uncheating")
                        self.logThread.printf("[系统消息]：头部姿态正常")

                if self.needStop:
                    camera.release()
                    p = QImage("./res/image_logo.png").scaled(150, 150)
                    self.changePixmap.emit(p)
                    break

            else:
                break

# 倒计时线程
class timeThread(QThread):
    timerCount = pyqtSignal(str)
    def __init__(self, parent=None):
        super(timeThread, self).__init__(parent)
        self.needStop = False
        self.hour = 1
        self.min = 59
        self.sec = 59

    def run(self):
        while True:
            for h in range(self.hour, 0, -1):
                for m in range(self.min, 0, -1):
                    for s in range(self.sec, 0 ,-1):
                        # print("\r%02d : %02d : %02d" %(h, m, s))
                        timer = "%02d : %02d : %02d" %(h, m , s)
                        self.timerCount.emit(timer)
                        time.sleep(1)
                        if s == 1:
                            self.sec = 59
                            break
                        if self.needStop:
                            return
                    if m == 1:
                        self.min = 59
                        break
                if h == 1:
                    break
            break

# 主界面程序
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.isStartButtonClicked = False

        from container import Ui_Container
        self.ui = Ui_Container()
        self.ui.setupUi(self)

        use_gpu = True
        resnet_model_path = "./model/ep000008-loss0.074-val_loss0.047.pd"
        resnet_class_file = 'dat/cheat_classes.txt'
        resnet_input_shape = [224, 224]
        self.resnet_decode = ResnetDecode(resnet_input_shape, resnet_model_path, resnet_class_file, use_gpu=use_gpu)
        self.face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")

        self.ui.btnReg.clicked.connect(self.onRegButtonPush)
        self.ui.btnStart.clicked.connect(self.onStartButtonPush)

        self.logThread = LogThread(self)
        self.logThread.trigger.connect(self.updateText)
        self.logThread.start()

        self.playThread = dpThread(self, logThread=self.logThread, resnetModel=self.resnet_decode, faceDetectModel=self.face_detector)
        self.playThread.changePixmap.connect(self.setImage)
        self.playThread.eventPush.connect(self.eventHandle)

        self.timerThread = timeThread(self)
        self.timerThread.timerCount.connect(self.updateDate)

        self.list = []
        


    def setImage(self, image):
        self.ui.cameraLabel.setPixmap(QPixmap.fromImage(image))

    def eventHandle(self, message):
        datetime = QDateTime.currentDateTime()
        singleData = "[%s]: %s" %(datetime.toString(Qt.ISODate), message)
        self.list.append(singleData)
        self.ui.eventListModel.setStringList(self.list)
        # self.ui.eventListView.setModel(self.eventListModel)

    def updateText(self, message):
        self.ui.logTextBrowser.append(message)

    def updateDate(self, message):
        self.ui.timeLabel.setText(message)

    def onRegButtonPush(self):
        self.logThread.printf("Reg Press")
        pass

    def onStartButtonPush(self):
        
        if not self.isStartButtonClicked:
            # START
            self.logThread.printf("[系统消息]：考试开始，点击“结束”按键结束考试。")
            self.isStartButtonClicked = True
            self.ui.btnStart.setText("结 束")
            self.playThread.needStop = False
            self.playThread.start()
            self.timerThread.needStop = False
            self.timerThread.start()
        else:
            # STOP
            self.logThread.printf("[系统消息]：考试结束。")
            self.isStartButtonClicked = False
            self.ui.btnStart.setText("开 始")
            self.timerThread.needStop = True
            self.playThread.needStop = True
            


if __name__=='__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.setWindowTitle("西普教育考试防作弊系统")
    w.setFixedSize(w.width(), w.height())
    w.show()
    sys.exit(app.exec_())