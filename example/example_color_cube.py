# -*- coding: utf-8 -*-
"""Color cube example for exploration of colors."""
import numpy as np
import matplotlib.pyplot as plt

import diagfig
from diagfig.utils import get_palette

# checking demo
diagfig.demo()
diagfig.demo("color_space", diagfig.IXORA)
diagfig.demo("color_space", diagfig.VIENOT)
diagfig.demo("color_space", diagfig.RUMINSKI)
#%% color cube in HSV
rgb = []
for i in np.linspace(0,1, 10):
    rgb.append(get_palette(val = i, n = 25)[:, :, :, np.newaxis])
rgbig = np.concatenate(rgb, axis = 3)

#%% checking the cube for a value of H, S, or V
fig, ax = plt.subplots(FigureClass=diagfig.FigureDiag, layout = "constrained")
ax.imshow(rgbig[:,:,:,7], aspect = "auto")
ax.axis("off")
diaged_fig = fig.diag()
plt.close(fig)

fig, ax = plt.subplots(FigureClass=diagfig.FigureDiag, layout = "constrained")
ax.imshow(np.swapaxes(rgbig[10,:,:,:], 2,1), aspect = "auto")
ax.axis("off")
diaged_fig = fig.diag()
plt.close(fig)

fig, ax = plt.subplots(FigureClass=diagfig.FigureDiag, layout = "constrained")
ax.imshow(np.swapaxes(rgbig[:,10,:,:], 1,2), aspect = "auto")
ax.axis("off")
diaged_fig = fig.diag()
plt.close(fig)
