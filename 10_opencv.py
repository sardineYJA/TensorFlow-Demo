import cv2
import numpy as np 

# 读入原图
image = cv2.imread('flower.jpg')
cv2.imshow('original', image)

# 低通滤波
kernel = np.array([[.11, .11, .11],
					[.11, .11, .11],
					[.11, .11, .11]])
rect = cv2.filter2D(image, -1, kernel)
cv2.imwrite('rect.jpg', rect)

# 高斯滤波
kernel = np.array([[1,4,7,4,1],
					[4,16,26,16,4],
					[7,26,41,26,7],
					[4,16,26,16,4],
					[1,4,7,4,1]]) / 273.0
gaussian = cv2.filter2D(image, -1, kernel)
cv2.imwrite('gaussian.jpg', gaussian)

# 锐化
kernel = np.array([[0, -2, 0],
					[-2, 9, -2],
					[0, -2, 0]])
sharpen = cv2.filter2D(image, -1, kernel)
cv2.imwrite('sharpen.jpg', sharpen)

# 边缘检测
kernel = np.array([[-1, -1, -1],
					[-1, 8, -1],
					[-1, -1, -1]])
edges = cv2.filter2D(image, -1, kernel)
cv2.imwrite('edges.jpg', edges)

# 浮雕
kernel = np.array([[-2, -2, -2, -2, 0],
					[-2, -2, -2, 0, 2],
					[-2, -2, 0, 2, 2],
					[-2, 0, 2, 2, 2],
					[0, 2, 2, 2, 2]])
emboss = cv2.filter2D(image, -1, kernel)
emboss = cv2.cvtColor(emboss, cv2.COLOR_BGR2GRAY)
cv2.imwrite('emboss.jpg', emboss)