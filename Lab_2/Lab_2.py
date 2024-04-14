import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.signal import convolve2d
import cv2

# Исходное изображение
filename = "C:\\Users\\Staval\\PycharmProjects\\IA_Labs\\Lab_1\\Images\\05_bridge.tif"
img = imread(filename)

# image_show(img)

# Простой градиент

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].set_title("Исходное изображение")

direction_x = np.array([[-1, 1]])
direction_y = np.array([[-1], [1]])

derivative_x = convolve2d(img, direction_x, mode='same', boundary='symm')
derivative_y = convolve2d(img, direction_y, mode='same', boundary='symm')

gradient = np.sqrt(derivative_x ** 2 + derivative_y ** 2)


ax[0, 1].imshow(derivative_x+128, cmap='gray', vmin=0, vmax=255)
ax[0, 1].set_title("Производная по x")

ax[0, 2].imshow(derivative_y+128, cmap='gray', vmin=0, vmax=255)
ax[0, 2].set_title("Производная по y")

ax[1, 0].imshow(gradient, cmap='gray', vmin=0, vmax=255)
ax[1, 0].set_title("Градиент")

ax[1, 1].hist(gradient.flatten(), 256, [0, 256])
ax[1, 1].set_title("Гистограмма градиента")

img_res = (gradient > 14) * 255

ax[1, 2].imshow(img_res, cmap='gray', vmin=0, vmax=255)
ax[1, 2].set_title("Контуры методом простого градиента")

plt.show()

edges = cv2.Canny(img, 50, 150)

plt.imshow(edges,  cmap='gray', vmin=0, vmax=255)
# Лапласиан

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].set_title("Исходное изображение")

window = np.array([[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]])

laplasian = convolve2d(img, window, mode='same', boundary='symm')

ax[0, 1].imshow(laplasian+128, cmap='gray', vmin=0, vmax=255)
ax[0, 1].set_title("Лапласиан")

ax[1, 0].hist(laplasian.flatten(), 256, [0, 256])
ax[1, 0].set_title("Гистограмма лапласиана")
img_res = (laplasian > 16) * 255

ax[1, 1].imshow(img_res, cmap='gray', vmin=0, vmax=255)
ax[1, 1].set_title("Контуры методом лапласиана")

plt.show()

# Прюитт

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].set_title("Исходное изображение")

window_1 = np.array([[-1 / 6, -1 / 6, -1 / 6],
                     [0, 0, 0],
                     [1 / 6, 1 / 6, 1 / 6]])

window_2 = np.array([[-1 / 6, 0, 1 / 6],
                     [-1 / 6, 0, 1 / 6],
                     [-1 / 6, 0, 1 / 6]])

s_1 = convolve2d(img, window_1, mode='same', boundary='symm')
s_2 = convolve2d(img, window_2, mode='same', boundary='symm')

pruitt = np.sqrt(s_1 ** 2 + s_2 ** 2)

ax[0, 1].imshow(s_1+128, cmap='gray', vmin=0, vmax=255)
ax[0, 1].set_title("S1")

ax[0, 2].imshow(s_2+128, cmap='gray', vmin=0, vmax=255)
ax[0, 2].set_title("S2")

ax[1, 0].imshow(pruitt, cmap='gray', vmin=0, vmax=255)
ax[1, 0].set_title("Прюитт")

ax[1, 1].hist(pruitt.flatten(), 256, [0, 256])
ax[1, 1].set_title("Гистограмма Прюитта")

img_res = (pruitt > 7) * 255

ax[1, 2].imshow(img_res, cmap='gray', vmin=0, vmax=255)
ax[1, 2].set_title("Контуры методом оператора Прюитта")

plt.show()

# Метод согласования лапласиана

fig, ax = plt.subplots(2, 2, figsize=(12, 7))
ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
ax[0, 0].set_title("Исходное изображение")

window = np.array([[2 / 3, -1 / 3, 2 / 3],
                   [-1 / 3, -4 / 3, -1 / 3],
                   [2 / 3, -1 / 3, 2 / 3]])

laplasian = np.abs(convolve2d(img, window, mode='same', boundary='symm'))  # Модуль для того, чтобы стало светлее, как я понял

ax[0, 1].imshow(laplasian, cmap='gray', vmin=0, vmax=255)
ax[0, 1].set_title("Лапласиан")

ax[1, 0].hist(laplasian.flatten(), 256, [0, 256])
ax[1, 0].set_title("Гистограмма лапласиана")

img_res = (laplasian > 16) * 255

ax[1, 1].imshow(img_res, cmap='gray', vmin=0, vmax=255)
ax[1, 1].set_title("Контуры методом согласования для лапласиана")

plt.show()
