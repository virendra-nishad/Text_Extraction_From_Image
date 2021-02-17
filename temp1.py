from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('image2.png', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low = hsv[0, 0]
high = hsv[0, 0]

mask_back = cv2.inRange(hsv, low, high)
mask_fore = cv2.bitwise_not(mask_back)

res = cv2.bitwise_and(hsv, hsv, mask=mask_fore)


plt.subplot(221), plt.imshow(hsv), plt.title('original')
plt.subplot(222), plt.imshow(res), plt.title('foreground')
#plt.subplot(233), plt.imshow(bac), plt.title('background')
plt.subplot(223), plt.imshow(mask_fore), plt.title('mask foreground')
plt.subplot(224), plt.imshow(mask_back), plt.title('mask background')
# plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

plt.show()
