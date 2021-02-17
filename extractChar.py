import os
import os.path
from cv2 import cv2
import glob
import imutils
import numpy as np

IMG_DES = "temp"


def pad_image(image):
    (h, w) = image.shape[:2]
    padW = int((140 - w)/2.0)
    padH = int((140 - h)/2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_CONSTANT)
    image = cv2.resize(image, (140, 140))
    return image

def extract(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = np.array([0, 100, 250])
    high = np.array([179, 255, 255])
    mask_fore = cv2.inRange(hsv, low, high)
    mask_back = cv2.bitwise_not(mask_fore)

    kernel = np.ones((11, 11), np.float32)/121
    fltr1_f_dil = cv2.dilate(mask_fore, kernel, iterations=1)
    fltr1_f_bor = cv2.bitwise_and(mask_back, mask_back, mask=fltr1_f_dil)

    contours, hierarchy = cv2.findContours(fltr1_f_bor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    i = 0

    save_path = os.path.join(os.getcwd(), IMG_DES)
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            p = os.path.join(save_path, "{}.png".format(str(i)))
            cv2.imwrite(p, pad_image(fltr1_f_bor[y:y+h, x:x+w]))
            i = i+1

