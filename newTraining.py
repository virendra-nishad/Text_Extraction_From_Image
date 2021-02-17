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
    # print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    image = cv2.imread(captcha_image_file)
    image = cv2.bitwise_not(image)

    # counter = 0
    save_path = os.path.join(IMG_DES, captcha_correct_text)
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for i in range(50):
        #i = j+1

        M = cv2.getRotationMatrix2D((cols/2, rows/2), i, 1)
        dst = cv2.warpAffine(image, M, (cols, rows))
        #dst.dtype = 'uint8'
        
        M1 = cv2.getRotationMatrix2D((cols/2, rows/2), i*(-1), 1)
        dst1 = cv2.warpAffine(image, M1, (cols, rows))
        #dst.dtype = 'uint8'

        p = os.path.join(save_path, "{}.png".format(str(i)))
        p1 = os.path.join(save_path, "{}.png".format(str(50 + i)))
        cv2.imwrite(p, dst)
        cv2.imwrite(p1, dst1)
    p1 = os.path.join(save_path, "{}.png".format(str(110)))
    p2 = os.path.join(save_path, "{}.png".format(str(111)))
    p3 = os.path.join(save_path, "{}.png".format(str(112)))
    cv2.imwrite(p1, image)
    cv2.imwrite(p2, image)
    cv2.imwrite(p3, image)
