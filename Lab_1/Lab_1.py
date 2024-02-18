import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread, imsave
from skimage import data, img_as_float
from skimage import exposure


matplotlib.rcParams['font.size'] = 8


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap='gray')
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    #ax_cdf.plot(bins, img_cdf, 'r')
    #ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


path = 'Images/01_apc.tif'
img = imread(path)
# Load an example image
# image_show(img)
# Equalization
img_eq = exposure.equalize_hist(img)
# image_show(img_eq)

# Display results
fig = plt.figure(figsize=(20, 10))
axes = np.zeros((2, 2), dtype=object)
axes[0, 0] = fig.add_subplot(2, 4, 1)
for i in range(1, 2):
    axes[0, i] = fig.add_subplot(2, 2, 1+i)
for i in range(0, 2):
    axes[1, i] = fig.add_subplot(2, 2, 3+i)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))


ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 1])
ax_img.set_title('Histogram equalization')


ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, y_max, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()