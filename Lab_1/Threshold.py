import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


text = "Images/01_apc.tif"
img = imread(text)
image_show(img)

fig, ax = plt.subplots(1, 1)
ax.hist(img.ravel(), bins=256, range=[0, 256])
ax.set_xlim(0, 256)
plt.show()

p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
image_show(img_rescale)
fig, ax = plt.subplots(1, 1)
ax.hist(img_rescale.ravel(), bins=256, range=[0, 256])
ax.set_xlim(0, 256)
plt.show()

selem = disk(100)
img_eq = rank.equalize(img, footprint=selem)
image_show(img_eq)
fig, ax = plt.subplots(1, 1)
ax.hist(img_eq.ravel(), bins=256, range=[0, 256])
ax.set_xlim(0, 256)
plt.show()


img_eq = exposure.equalize_hist(img)
image_show(img_eq)
fig, ax = plt.subplots(1, 1)
ax.hist(img_eq.ravel(), bins=256, range=[0, 256])
ax.set_xlim(0, 256)
plt.show()


text_segmented = img > 100
# text_segmented = filters.threshold_local(img, block_size=21, offset=10)
image_show(text_segmented);
fig, ax = plt.subplots(1, 1)
ax.hist(text_segmented.ravel(), bins=256, range=[0, 1])
ax.set_xlim(0, 1)
plt.show()
