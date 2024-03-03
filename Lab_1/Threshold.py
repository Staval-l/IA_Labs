from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def threshold_processing(image, threshold):
    result = (image > threshold).astype(np.uint8) * 255

    plt.figure(figsize=(16, 9))
    plt.suptitle("Пороговая обработка", fontsize=15, fontweight='bold')

    plt.subplot(2, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 3, 2)
    plt.title("Изображение после пороговой обработки")
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 3, 4)
    plt.title("Исходная гистограмма")
    plt.hist(image.flatten(), bins=256)

    plt.subplot(2, 3, 5)
    plt.title("Гистограмма после пороговой обработки")
    plt.hist(result.flatten(), bins=256)

    x = np.linspace(0, 255, 256)
    y = (x > threshold).astype(np.uint8) * 255
    plt.subplot(1, 3, 3)
    plt.title("График функции поэлементного преобразования")
    plt.plot(x, y)

    plt.show()


def image_contrasting(image, gmin=0, gmax=255):
    fmin = np.min(image)
    fmax = np.max(image)
    a = (gmax - gmin) / (fmax - fmin)
    b = (gmin * fmax - gmax * fmin) / (fmax - fmin)
    result = (image * a + b).astype(np.uint8)

    plt.figure(figsize=(16, 9))
    plt.suptitle("Линейное контрастирование", fontsize=15, fontweight='bold')

    plt.subplot(2, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 3, 2)
    plt.title("Изображение после контрастирования")
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 3, 4)
    plt.title("Исходная гистограмма")
    plt.hist(image.flatten(), bins=256)

    plt.subplot(2, 3, 5)
    plt.title("Гистограмма после контрастирования")
    plt.hist(result.flatten(), bins=256)

    x = np.linspace(0, 255, 256)
    y = (x * a + b).astype(np.int16)
    y[y < 0] = 0
    y[y > 255] = 255
    plt.subplot(1, 3, 3)
    plt.title("График функции поэлементного преобразования")
    plt.plot(x, y)

    plt.show()


def image_equalization(image, gmin=0, gmax=255):
    hist_source, bins = np.histogram(image.flatten(), 256, (0, 256))

    cdf = hist_source.cumsum()
    cdf_normalized_source = cdf / cdf.max()
    g_f = (gmax - gmin) * cdf_normalized_source + gmin

    result = g_f[image].astype(np.uint8)

    hist_result, _ = np.histogram(result.flatten(), 256, (0, 256))
    cdf_result = hist_result.cumsum()
    cdf_normalized_result = cdf_result / cdf_result.max()

    lib_res = cv.equalizeHist(image)

    plt.figure(figsize=(16, 9))
    plt.suptitle("Эквализация", fontsize=15, fontweight='bold')

    plt.subplot(2, 3, 1)
    plt.title("Исходное изображение")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 3, 2)
    plt.title("Изображение после эквализации")
    plt.imshow(result, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 3, 3)
    plt.title("Изображение после эквализации (opencv)")
    plt.imshow(lib_res, cmap='gray', vmin=0, vmax=255)

    plt.subplot(2, 3, 4)
    plt.title("Исходная гистограмма")
    plt.hist(image.flatten(), bins=bins)

    plt.subplot(2, 3, 5)
    plt.title("Гистограмма после эквализации")
    plt.hist(result.flatten(), bins=256)

    plt.subplot(2, 3, 6)
    plt.title("Гистограмма после эквализации (opencv)")
    plt.hist(lib_res.flatten(), bins=256)

    plt.show()

    plt.figure(figsize=(16, 9))
    plt.suptitle("Эквализация", fontsize=19, fontweight='bold')

    plt.subplot(2, 2, 1)
    plt.title("График интегральной функции распределения яркости (до обработки)")
    plt.plot(cdf_normalized_source)

    plt.subplot(2, 2, 3)
    plt.title("График интегральной функции распределения яркости (после обработки)")
    plt.plot(cdf_normalized_result)

    x = np.linspace(0, 255, 256, dtype=int)
    y = g_f[x].astype(np.uint8)
    plt.subplot(1, 3, 3)
    plt.title("График функции поэлементного преобразования")
    plt.plot(x, y)

    plt.show()

    print(f"Гистограмма исходного изображения: {hist_source}")
    print(f"Границы интервалов гистограммы: {bins}")


def main():
    path_to_image = 'Images/07_elaine.tif'
    threshold = 120
    image = imread(path_to_image)

    threshold_processing(image, threshold)
    image_contrasting(image)
    image_equalization(image)


if __name__ == "__main__":
    main()
