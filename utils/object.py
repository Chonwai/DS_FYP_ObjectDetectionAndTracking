import base64
from io import BytesIO
import cv2
from utils.utils import Utils

class ObjectUtils:
    @staticmethod
    def writeCircleStatusOnFrame(frame, x, y, area):
        if Utils.isInside(x, y, area) == True:
            cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        elif Utils.isInside(x, y, area) == False:
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)


