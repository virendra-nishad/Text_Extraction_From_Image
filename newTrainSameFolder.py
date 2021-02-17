import os
import os.path
from cv2 import cv2
import glob
import imutils


IMG_SRC = "reference"
IMG_DES = "referenceNew"
rows = 140
cols = 140

captcha_image_files = glob.glob(os.path.join(IMG_SRC, "*"))
counts = {}

for (i, captcha_image_file) in enumerate(captcha_image_files):
    filename = os.path.basename(captcha_image_file)
    char_name = os.path.splitext(filename)[0]

    image = cv2.imread(captcha_image_file)
    image = cv2.bitwise_not(image)

    save_path = IMG_DES
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for j in range(4):
        i = j+1

        M = cv2.getRotationMatrix2D((cols/2, rows/2), i*10, 1)
        dst = cv2.warpAffine(image, M, (cols, rows))

        M1 = cv2.getRotationMatrix2D((cols/2, rows/2), i*10*(-1), 1)
        dst1 = cv2.warpAffine(image, M1, (cols, rows))

        p = os.path.join(save_path, "{}.png".format(char_name + str(i)))
        p1 = os.path.join(save_path, "{}.png".format(char_name + str(4 + i)))
        cv2.imwrite(p, dst)
        cv2.imwrite(p1, dst1)
    p = os.path.join(save_path, "{}.png".format(char_name + str(0)))
    cv2.imwrite(p, image)
