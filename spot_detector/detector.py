import cv2
import colorsys
import numpy as np

class LocalDetector(object):
    """
    Detect the empty spots in the local, instance-centric view
    """
    def __init__(self, spot_color_rgb):
        """
        Instantiate the local detector object

        spot_color_rgb: 3-element tuple, rgb values of empty spot in the input image
        """
        h, s, v = colorsys.rgb_to_hsv( *np.array(spot_color_rgb)/255 )
        self.hsv_lower = (h*180-10, s*255-30, v*255-30)
        self.hsv_upper = (h*180+10, s*255, v*255)

    def spots_mask(self, img):
        """
        apply a color filter for the open spots in the image

        img: PIL image
        """
        hsv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_img, self.hsv_lower, self.hsv_upper)

        res = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)
        
        return res

    def detect(self, img, area_thres=600):
        """
        detect rectangular spots

        area_thres: the area threshold of the detected rectangle. Only the ones above this threshold is considered. In pixel coordinate
        """
        res = self.spots_mask(img)
        _, _, gray = cv2.split(res)

        cnts, _ = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            # If this polygon has 4 edges
            if len(approx) == 4:
                rect = cv2.minAreaRect(c)
                # If the area is greater than the threshold
                if rect[1][0] * rect[1][1] > area_thres:
                    boxes.append(rect)

        return boxes