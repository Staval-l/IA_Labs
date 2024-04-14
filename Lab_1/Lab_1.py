# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# from skimage.io import imread, imsave
# from skimage import data, img_as_float
# from skimage import exposure
#
#
# matplotlib.rcParams['font.size'] = 8
#
#
# def image_show(image, nrows=1, ncols=1, cmap='gray'):
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
#     ax.imshow(image, cmap='gray')
#     ax.axis('off')
#     return fig, ax
#
#
# def plot_img_and_hist(image, axes, bins=256):
#     """Plot an image along with its histogram and cumulative histogram.
#
#     """
#     image = img_as_float(image)
#     ax_img, ax_hist = axes
#     ax_cdf = ax_hist.twinx()
#
#     # Display image
#     ax_img.imshow(image, cmap='gray')
#     ax_img.set_axis_off()
#
#     # Display histogram
#     ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
#     ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
#     ax_hist.set_xlabel('Pixel intensity')
#     ax_hist.set_xlim(0, 1)
#     ax_hist.set_yticks([])
#
#     # Display cumulative distribution
#     img_cdf, bins = exposure.cumulative_distribution(image, bins)
#     #ax_cdf.plot(bins, img_cdf, 'r')
#     #ax_cdf.set_yticks([])
#
#     return ax_img, ax_hist, ax_cdf
#
#
# path = 'Images/01_apc.tif'
# img = imread(path)
# # Load an example image
# # image_show(img)
# # Equalization
# img_eq = exposure.equalize_hist(img)
# # image_show(img_eq)
#
# # Display results
# fig = plt.figure(figsize=(20, 10))
# axes = np.zeros((2, 2), dtype=object)
# axes[0, 0] = fig.add_subplot(2, 4, 1)
# for i in range(1, 2):
#     axes[0, i] = fig.add_subplot(2, 2, 1+i)
# for i in range(0, 2):
#     axes[1, i] = fig.add_subplot(2, 2, 3+i)
#
# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
# ax_img.set_title('Low contrast image')
#
# y_min, y_max = ax_hist.get_ylim()
# ax_hist.set_ylabel('Number of pixels')
# ax_hist.set_yticks(np.linspace(0, y_max, 5))
#
#
# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 1])
# ax_img.set_title('Histogram equalization')
#
#
# ax_cdf.set_ylabel('Fraction of total intensity')
# ax_cdf.set_yticks(np.linspace(0, y_max, 5))
#
# # prevent overlap of y-axis labels
# fig.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def threshold_processing(image, threshold):
    result = (image > threshold).astype(np.uint8) * 255

    plt.figure(figsize=(16, 9))
    plt.suptitle("Пороговая обработка", fontsize=19, fontweight='bold')

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
    plt.suptitle("Контрастирование", fontsize=19, fontweight='bold')

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
    plt.suptitle("Эквализация", fontsize=19, fontweight='bold')

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
    print(f"Границы интервалов этой гистограммы: {bins}")


def main():
    path_to_image = 'Images/07_elaine.tif'
    threshold = 127
    image = plt.imread(path_to_image)

    threshold_processing(image, threshold)
    image_contrasting(image)
    image_equalization(image)


if __name__ == "__main__":
    main()
