import base64
from io import BytesIO
import cv2
from utils.area import AreaUtils
import json
import os


class Utils:
    """Calculate the Coordinate of BBox's Bottom Center Point

        Parameters
        ----------
        x1 : int
            bbox[0]
        y1 : int
            bbox[1]
        x2 : int
            bbox[2]
        y2 : int
            bbox[3]

        Returns
        -------
        (int, int)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
    @staticmethod
    def calculateBottomCenterCoordinate(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        return [int(x), int(y2)]

    @staticmethod
    def isInside(new_x=0, new_y=0, area=None):
        area_poly = AreaUtils.getPolygonShape(
            json.loads(str(area)))

        if not area_poly:
            return False

        nvert = len(area_poly)
        vertx = []
        verty = []
        testx = new_x
        testy = new_y
        for item in area_poly:
            vertx.append(item[0])
            verty.append(item[1])

        j = nvert - 1
        res = False
        for i in range(nvert):
            if (verty[j] - verty[i]) == 0:
                j = i
                continue
            x = (vertx[j] - vertx[i]) * (testy - verty[i]) / \
                (verty[j] - verty[i]) + vertx[i]
            if ((verty[i] > testy) != (verty[j] > testy)) and (testx < x):
                res = not res
            j = i

        return res

    @staticmethod
    def imageToBase64(image):
        retval, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer)
        return jpg_as_text
