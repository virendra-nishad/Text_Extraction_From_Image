import os
import os.path
from cv2 import cv2
import glob
import imutils
from copy1 import pad_image
import shutil
from imutils import paths
import numpy as np


CAPTCHA_IMAGE_FOLDER = "train"
OUTPUT_FOLDER = "out"

captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

counts = {}
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    image = cv2.imread(captcha_image_file)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = np.array([0, 100, 250])
    high = np.array([179, 255, 255])
    mask_fore = cv2.inRange(hsv, low, high)
    # mask_back = cv2.bitwise_not(mask_fore)

    # kernel = np.ones((11, 11), np.float32)/121
    # fltr1_f_dil = cv2.dilate(mask_fore, kernel, iterations=1)
    # fltr1_f_bor = cv2.bitwise_and(mask_back, mask_back, mask=fltr1_f_dil)

    contours, hierarchy = cv2.findContours(
        mask_fore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    save_path = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    for cnt, letter in zip(contours, captcha_correct_text):
        save_path = os.path.join(OUTPUT_FOLDER, letter)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 50:
            count = counts.get(letter, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, pad_image(mask_fore[y:y+h, x:x+w]))
            counts[letter] = count + 1
