import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


def image_show(image, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


# Дефолт
text = "Images/09_lena2.tif"
img = imread(text)
image_show(img)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf_o = hist.cumsum()
cdf_normalized = cdf_o * float(hist.max()) / cdf_o.max()

fig, ax = plt.subplots(1, 1)
ax.hist(img.ravel(), bins=256, range=[0, 256])
ax.plot(cdf_normalized, color='red')
ax.set_xlim(0, 256)
plt.show()

# Линейное контрастирование
image = imread(text)
min_val = np.min(image)
max_val = np.max(image)
a = 255 / (max_val - min_val)
b = -1 * (255 * min_val) / (max_val - min_val)
image = a * image + b
image_show(image)

fig, ax = plt.subplots(1, 1)
ax.hist(image.ravel(), bins=256, range=[0, 256])
ax.set_xlim(0, 256)
plt.show()

# Эквализация гистограммы
img_eq = imread(text)
hist, bins = np.histogram(img_eq.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img_eq = cdf[img_eq]
image_show(img_eq)

fig, ax = plt.subplots(1, 1)
ax.plot(cdf_normalized, color='red')
ax.hist(img_eq.ravel(), bins=256, range=[0, 256])
ax.set_xlim(0, 256)
plt.show()

# Пороговая обработка
img_t = (img > 100).astype('uint8')
image_show(img_t)
fig, ax = plt.subplots(1, 1)
ax.hist(img_t.ravel(), bins=256, range=[0, 1])
ax.set_xlim(0, 1)
plt.show()
