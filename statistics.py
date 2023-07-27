import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import skimage.data
    import skimage.io
    from skimage import img_as_float
    from skimage.transform import resize
except ImportError:
    raise ImportError("This example requires scikit-image")

from FFST import (scalesShearsAndSpectra,
                  shearletTransformSpect)

def add_cbar(im, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.05)
    plt.colorbar(im, cax=cax)

def data_weighting(conf, filter, scale):
    J = 0
    for i in range(scale):
        J += 2**(i+2)
    conf[..., J] = conf[..., J] / np.linalg.norm(filter[..., J])

img = 'paper_img.jpg'
cmap = plt.cm.Greys
scale = 2

Max = []
Min = []
Mean = []
Std = []

X = skimage.io.imread(img, as_gray=True)
X = img_as_float(X)
X = resize(X, (512, 512))

# compute shearlet transform
ST, Psi = shearletTransformSpect(X, numOfScales=scale)  ##  ST, Psi (256, 256, 61)
i = 13

for idx in range(1, i):
    #if idx != 0:
    data_weighting(ST, Psi, scale)
    Max.append(np.max(ST[..., idx]))
    Min.append(np.min(ST[..., idx]))
    Mean.append(np.mean(ST[..., idx]))
    Std.append(np.std(ST[..., idx]))

np.save('low_frequency.npy', ST[..., 0])
np.save('high_frequency.npy', ST[..., 1:i])
plt.figure(1, figsize=(8, 2))

ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
ax.get_yaxis().get_offset_text().set(va='bottom', ha='left', fontproperties='Times New Roman', fontsize=13, fontweight='bold')
ax.yaxis.get_offset_text().set_fontsize(13)

plt.scatter(range(1, i), Std)
plt.xlabel('Shearlet Index', fontproperties='Times New Roman', fontsize=13, labelpad=1, fontweight='bold')
plt.ylabel('Std', fontproperties='Times New Roman', fontsize=13, fontweight='bold')
plt.xticks(range(1, i), fontproperties='Times New Roman', size=13)
plt.yticks(fontproperties='Times New Roman', size=13)

plt.savefig(r"Std.jpg", dpi= 500, bbox_inches='tight')
plt.show()
