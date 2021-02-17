from keras.models import load_model
from imutils import paths
import numpy as np
import imutils
from cv2 import cv2
import pickle
from extractChar import pad_image
import os



MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "test1"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)

# Grab some random CAPTCHA images to test against.
# In the real world, you'd replace this section with code to grab a real
# CAPTCHA image from a live website.
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
# captcha_image_files = np.random.choice(
#     captcha_image_files, size=(3,), replace=False)

# loop over the image paths
for image_file in captcha_image_files:
    # Load the image and convert it to grayscale
    print(image_file)
    image = cv2.imread(image_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.imread(image_file)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = np.array([0, 100, 250])
    high = np.array([179, 255, 255])
    mask_fore = cv2.inRange(hsv, low, high)
    mask_back = cv2.bitwise_not(mask_fore)

    kernel = np.ones((11, 11), np.float32)/121
    fltr1_f_dil = cv2.dilate(mask_fore, kernel, iterations=1)
    fltr1_f_bor = cv2.bitwise_and(mask_back, mask_back, mask=fltr1_f_dil)

    contours, hierarchy = cv2.findContours(
        fltr1_f_bor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # save_path = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    # if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    # for cnt, letter in zip(contours, captcha_correct_text):
    #     save_path = os.path.join(OUTPUT_FOLDER, letter)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     if w > 50 and h > 50:
    #         count = counts.get(letter, 1)
    #         p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
    #         cv2.imwrite(p, pad_image(fltr1_f_bor[y:y+h, x:x+w]))
    #         counts[letter] = count + 1
    # letter_image_regions = []

    predictions = []

    # loop over the lektters
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        im = pad_image(fltr1_f_bor[y:y+h, x:x+w])
        thresh = 127
        im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)[1]
        im = np.expand_dims(im, axis=2)
        im = np.expand_dims(im, axis=0)
        prediction = model.predict(im)

        # Convert the one-hot-encoded prediction back to a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    # Show the annotated image
    # cv2.imshow("Output", output)
    # cv2.waitKey()
