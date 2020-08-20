from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
import cv2
import time


class Thread(QThread):#采用线程来播放视频
    changePixmap = pyqtSignal(QImage)

    def run(self):
        camera = cv2.VideoCapture(0)
        while (camera.isOpened()==True):
            res, frame = camera.read()
            if res:
                frame = cv2.resize(frame, (700, 500))
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                p = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], rgbImage.shape[1]*3, QImage.Format_RGB888)
                self.changePixmap.emit(p)
                time.sleep(0.01) #控制视频播放的速度
            else:
                break

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.isStartButtonClicked = False

        from container import Ui_Container
        self.ui = Ui_Container()
        self.ui.setupUi(self)

        self.ui.btnReg.clicked.connect(self.onRegButtonPush)
        self.ui.btnStart.clicked.connect(self.onStartButtonPush)

        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()

    def setImage(self, image):
        self.ui.cameraLabel.setPixmap(QPixmap.fromImage(image))

    def onRegButtonPush(self):
        print("Reg Press")
        pass

    def onStartButtonPush(self):
        print("Start Press")
        if not self.isStartButtonClicked:
            # START
            self.isStartButtonClicked = True
            self.ui.btnStart.setText("结 束")
        else:
            # STOP
            self.isStartButtonClicked = False
            self.ui.btnStart.setText("开 始")


if __name__=='__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.setWindowTitle("西普教育考试防作弊系统")
    w.setFixedSize(w.width(), w.height())
    w.show()
    sys.exit(app.exec_())