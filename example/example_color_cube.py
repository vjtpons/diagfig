# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import colorsys

import numpy as np
import matplotlib.pyplot as plt

import diagfig
from diagfig.utils import (figure_to_rgba_array,
                             rgb2gray_digital,
                             rgb2gray_human_eye,
                             simulate_colorblindness, 
                             rgb_to_lms,
                             lms_to_rgb,
                             get_palette)
diagfig.demo()
diagfig.demo("color_space", diagfig.IXORA)
diagfig.demo("color_space", diagfig.VIENOT)
diagfig.demo("color_space", diagfig.RUMINSKI)
#%%

rgb = []
for i in np.linspace(0,1, 10):
    rgb.append(get_palette(val = i, n = 25)[:, :, :, np.newaxis])
    # fig, ax = plt.subplots(FigureClass=diagfig_dev.FigureDiag, layout = "constrained")
    # ax.imshow(rgb)
    # ax.axis("off")
    # diaged_fig = fig.diag()
    # plt.close(fig)
rgbig = np.concatenate(rgb, axis = 3)

#%%
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







