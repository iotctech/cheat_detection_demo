import sys
import numpy as np
import cv2

class ImageHandle(object):

    def __init__(self):
        self.camera = cv2.VideoCapture(0)

    def frameHandle(self):
        res, frame = self.camera.read()
        show = cv2.resize(frame, (700, 500))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        return res, show