import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter


def Get_SKO(result, img):
    eps = 0
    for i in range(128):
        for j in range(128):
            eps += (result[i][j] - img[i][j]) ** 2
    eps /= 16384
    return eps


def task1(img):
    disp = np.var(img)

    print("Сигнал/шум = 1")
    print(f"Дисперсия изображения: {disp}")

    gauss = np.random.normal(0, np.sqrt(disp), img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1])

    # impulse = random_noise(img, mode='s&p', amount=0.1)

    noisy = img + gauss

    window = np.array([[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]])

    result_linear = convolve2d(noisy, window, mode='same', boundary='symm')

    eps = Get_SKO(result_linear, img)

    epsilon = Get_SKO(noisy, img)

    print(f"Дисперсия восстановления (линейный): {eps}")

    print(f"Коэф снижения шума (линейный): {eps / epsilon}")

    result_median = median_filter(noisy, size=5)

    eps = Get_SKO(result_median, img)

    print(f"Дисперсия восстановления (медианный): {eps}")

    print(f"Коэф снижения шума (медианный): {eps / epsilon}")

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[0, 0].set_title("Исходное изображение")

    ax[0, 1].imshow(gauss + 128, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].set_title("Шум (сигнал/шум = 1)")

    ax[0, 2].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    ax[0, 2].set_title("Зашумленное изображение")

    ax[1, 0].imshow(result_median, cmap='gray', vmin=0, vmax=255)
    ax[1, 0].set_title("Медианный")

    ax[1, 1].imshow(result_linear, cmap='gray', vmin=0, vmax=255)
    ax[1, 1].set_title("Линейный сглаживающий")

    plt.show()


def task2(img):
    disp = np.var(img)

    print("\nСигнал/шум = 10")
    print(f"Дисперсия изображения: {disp / 10}")

    gauss = np.random.normal(0, np.sqrt(disp / 10), img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1])

    noisy = img + gauss

    window = np.array([[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]])

    result_linear = convolve2d(noisy, window, mode='same', boundary='symm')

    eps = Get_SKO(result_linear, img)

    epsilon = Get_SKO(noisy, img)

    print(f"Дисперсия восстановления (линейный): {eps}")

    print(f"Коэф снижения шума (линейный): {eps / epsilon}")

    result_median = median_filter(noisy, size=5)

    eps = Get_SKO(result_median, img)

    print(f"Дисперсия восстановления (медианный): {eps}")

    print(f"Коэф снижения шума (медианный): {eps / epsilon}")

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[0, 0].set_title("Исходное изображение")

    ax[0, 1].imshow(gauss + 128, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].set_title("Шум (сигнал/шум = 10)")

    ax[0, 2].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    ax[0, 2].set_title("Зашумленное изображение")

    ax[1, 0].imshow(result_median, cmap='gray', vmin=0, vmax=255)
    ax[1, 0].set_title("Медианный")

    ax[1, 1].imshow(result_linear, cmap='gray', vmin=0, vmax=255)
    ax[1, 1].set_title("Линейный сглаживающий")

    plt.show()


def task3(img):
    disp = np.var(img)

    print("\np = 0.1")
    print(f"Дисперсия изображения: {disp}")

    normalized_img = img / 255
    impulse = np.zeros((128, 128))
    noisy = random_noise(normalized_img, mode='s&p', amount=0.1)
    noisy *= 255
    # impulse += 128
    # impulse=random_noise(impulse, mode='s&p', amount=0.3)
    impulse = np.copy(noisy)
    impulse[impulse == 96] = 128
    impulse[impulse == 160] = 128

    window = np.array([[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]])

    result_linear = convolve2d(noisy, window, mode='same', boundary='symm')

    eps = Get_SKO(result_linear, img)

    epsilon = Get_SKO(noisy, img)

    print(f"Дисперсия восстановления (линейный): {eps}")

    print(f"Коэф снижения шума (линейный): {eps / epsilon}")

    result_median = median_filter(noisy, size=5)

    eps = Get_SKO(result_median, img)

    print(f"Дисперсия восстановления (медианный): {eps}")

    print(f"Коэф снижения шума (медианный): {eps / epsilon}")

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[0, 0].set_title("Исходное изображение")

    ax[0, 1].imshow(impulse, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].set_title("Шум (p=0.1)")

    ax[0, 2].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    ax[0, 2].set_title("Зашумленное изображение")

    ax[1, 0].imshow(result_median, cmap='gray', vmin=0, vmax=255)
    ax[1, 0].set_title("Медианный")

    ax[1, 1].imshow(result_linear, cmap='gray', vmin=0, vmax=255)
    ax[1, 1].set_title("Линейный сглаживающий")

    plt.show()


def task4(img):
    disp = np.var(img)

    print("\np = 0.3")
    print(f"Дисперсия изображения: {disp}")

    normalized_img = img / 255
    impulse = np.zeros((128, 128))
    noisy = random_noise(normalized_img, mode='s&p', amount=0.3)
    noisy *= 255
    #impulse += 128
    #impulse=random_noise(impulse, mode='s&p', amount=0.3)
    impulse = np.copy(noisy)
    impulse[impulse == 96] = 128
    impulse[impulse == 160] = 128

    window = np.array([[1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9],
                       [1 / 9, 1 / 9, 1 / 9]])

    result_linear = convolve2d(noisy, window, mode='same', boundary='symm')

    eps = Get_SKO(result_linear, img)

    epsilon = Get_SKO(noisy, img)

    print(f"Дисперсия восстановления (линейный): {eps}")

    print(f"Коэф снижения шума (линейный): {eps / epsilon}")

    result_median = median_filter(noisy, size=5)

    eps = Get_SKO(result_median, img)

    print(f"Дисперсия восстановления (медианный): {eps}")

    print(f"Коэф снижения шума (медианный): {eps / epsilon}")

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))

    ax[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
    ax[0, 0].set_title("Исходное изображение")

    ax[0, 1].imshow(impulse, cmap='gray', vmin=0, vmax=255)
    ax[0, 1].set_title("Шум (p=0.3)")

    ax[0, 2].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    ax[0, 2].set_title("Зашумленное изображение")

    ax[1, 0].imshow(result_median, cmap='gray', vmin=0, vmax=255)
    ax[1, 0].set_title("Медианный")

    ax[1, 1].imshow(result_linear, cmap='gray', vmin=0, vmax=255)
    ax[1, 1].set_title("Линейный сглаживающий")

    plt.show()


img = np.zeros((128, 128))

img[:64, :64] = 96
img[:64, 64:128] = 160

img[64:128, :64] = 160
img[64:128, 64:128] = 96

task1(img)
task2(img)
task3(img)
task4(img)
