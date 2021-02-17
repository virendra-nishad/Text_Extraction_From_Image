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
OUTPUT_FOLDER = "test"

captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))

count = 0
code = []
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    code.append(captcha_correct_text)
    image = cv2.imread(captcha_image_file)

    save_path = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    if not os.path.exists(save_path):
            os.makedirs(save_path)

    p = os.path.join(save_path, "image{}.png".format(str(count)))
    cv2.imwrite(p, image)
    count = count + 1

with open('test/codes.txt', 'w') as f:
    for item in code:
        f.write("%s\n" % item)
f.close()
