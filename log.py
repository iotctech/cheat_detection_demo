from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

# 线程将打印输出到QTextBrowser
class LogThread(QThread):
    trigger = pyqtSignal(str)

    def __init__(self, parent=None):
        super(LogThread, self).__init__(parent)

    def printf(self, message):
        self.trigger.emit(message)