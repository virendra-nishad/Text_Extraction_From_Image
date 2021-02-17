from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('image2.png', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

low = np.array([0, 100, 250])
high = np.array([179, 255, 255])

mask_fore = cv2.inRange(hsv, low, high)
mask_back = cv2.bitwise_not(mask_fore)
# res = cv2.bitwise_and(hsv, hsv, mask=mask_fore)
bac = cv2.bitwise_and(hsv, hsv, mask=mask_back)

# low = np.array([0, 0, 0])
# high = np.array([0, 0, 0])
# fltr1_b = cv2.inRange(bac, low, high)
# # kernel = np.ones((5, 5), np.float32)/25
# fltr1_f = cv2.bitwise_not(fltr1_b)
# # fltr1_f = cv2.filter2D(fltr1_f, -1, kernel)

# image dilation
kernel = np.ones((9, 9), np.float32)/81
fltr1_f_dil = cv2.dilate(mask_fore, kernel, iterations=1)
# fltr1_f_ero = cv2.erode(mask_fore, kernel, iterations=1)
# fltr1_f_ero = cv2.bitwise_not(fltr1_f_ero)
#fltr1_f_dil = cv2.filter2D(fltr1_f_dil, -1, kernel)

fltr1_f_bor = cv2.bitwise_and(mask_back, mask_back, mask=fltr1_f_dil)
# kernel = np.ones((3, 3), np.float32)/9
# fltr1_f_bor = cv2.filter2D(fltr1_f_bor, -1, kernel)
txt = cv2.bitwise_and(img, img, mask=fltr1_f_bor)

plt.subplot(321), plt.imshow(img), plt.title('original')
# plt.subplot(332), plt.imshow(h), plt.title('h')
# plt.subplot(333), plt.imshow(s), plt.title('s')
# plt.subplot(334), plt.imshow(v), plt.title('v')
plt.subplot(322), plt.imshow(mask_back), plt.title('mask_back')
plt.subplot(323), plt.imshow(mask_fore), plt.title('mask_fore')
plt.subplot(324), plt.imshow(fltr1_f_bor), plt.title('fltr1_f_bor')
#plt.subplot(335), plt.imshow(bac), plt.title('bac')
#plt.subplot(336), plt.imshow(fltr1_b), plt.title('fltr1_b')
#plt.subplot(337), plt.imshow(fltr1_f), plt.title('fltr1_f')
plt.subplot(325), plt.imshow(txt), plt.title('txt')
plt.subplot(326), plt.imshow(fltr1_f_dil), plt.title('fltr1_f_ero')
#plt.subplot(4413), plt.imshow(fltr1_f_bor), plt.title('border')

plt.show()
