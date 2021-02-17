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
bac = cv2.bitwise_and(hsv, hsv, mask=mask_back)

kernel = np.ones((9, 9), np.float32)/81
fltr1_f_dil = cv2.dilate(mask_fore, kernel, iterations=1)

fltr1_f_bor = cv2.bitwise_and(mask_back, mask_back, mask=fltr1_f_dil)

txt = cv2.bitwise_and(img, img, mask=fltr1_f_bor)

contours, hierarchy = cv2.findContours(mask_fore, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
# 	x, y, w, h = cv2.boundingRect(cnt)
# 	#bound the images
# 	cv2.rectangle(txt, (x, y), (x+w, y+h), (0, 255, 0), 3)
i = 0
for cnt in contours:
	x, y, w, h = cv2.boundingRect(cnt)
	#following if statement is to ignore the noises and save the images which are of normal size(character)
	#In order to write more general code, than specifying the dimensions as 100,
	# number of characters should be divided by word dimension
	if w > 50 and h > 50:
		#save individual images
		cv2.imwrite(str(i)+".jpg", mask_fore[y:y+h, x:x+w])
		i = i+1
# cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
# cv2.imshow('BindingBox', txt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.subplot(321), plt.imshow(img), plt.title('original')
plt.subplot(322), plt.imshow(mask_back), plt.title('mask_back')
plt.subplot(323), plt.imshow(mask_fore), plt.title('mask_fore')
plt.subplot(324), plt.imshow(fltr1_f_bor), plt.title('fltr1_f_bor')
plt.subplot(325), plt.imshow(txt), plt.title('txt')
plt.subplot(326), plt.imshow(fltr1_f_dil), plt.title('fltr1_f_ero')
plt.show()
