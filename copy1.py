from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
from cv2 import cv2
import pickle
import os
import shutil


# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# filenames: a list of strings containing filenames of images

# OUTPUT CONVENTION
# The method must return a numpy array (not numpy matrix or scipy matrix) and a list of strings.
# Make sure that the length of the array and the list is the same as the number of filenames that
# were given. The evaluation code may give unexpected results if this convention is not followed.

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
#CAPTCHA_IMAGE_FOLDER = "test1"

with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILENAME)


def pad_image(image):
    (h, w) = image.shape[:2]
    padW = int((140 - w)/2.0)
    padH = int((140 - h)/2.0)
    image = cv2.copyMakeBorder(
        image, padH, padH, padW, padW, cv2.BORDER_CONSTANT)
    image = cv2.resize(image, (140, 140))
    return image

def decaptcha(filenames):
    captcha_image_files = filenames
    numChars = 3 * np.ones((len(filenames),))
    count = 0
    codes = []
    for image_file in captcha_image_files:
        image = cv2.imread(image_file)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low = np.array([0, 100, 250])
        high = np.array([179, 255, 255])
        mask_fore = cv2.inRange(hsv, low, high)
        #mask_back = cv2.bitwise_not(mask_fore)

        # kernel = np.ones((11, 11), np.float32)/121
        # fltr1_f_dil = cv2.dilate(mask_fore, kernel, iterations=1)
        # fltr1_f_bor = cv2.bitwise_and(mask_back, mask_back, mask=fltr1_f_dil)

        contours, hierarchy = cv2.findContours(
            mask_fore, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        predictions = []
        temp = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            im = pad_image(mask_fore[y:y+h, x:x+w])
            thresh = 127
            im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
            im = np.expand_dims(im, axis=2)
            im = np.expand_dims(im, axis=0)
            prediction = model.predict(im)
            letter = lb.inverse_transform(prediction)[0]
            predictions.append(letter)
            temp = temp + 1
        numChars[count] = temp
        count = count + 1
        captcha_text = "".join(predictions)
        # print(count)
        # print(temp)
        # print(captcha_text)
        codes.append(captcha_text)

    return (numChars, codes)
