# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:09:17 2024

@author: vjtpons
"""

import numpy as np
import matplotlib.pyplot as plt

import diagfig
from diagfig.helpers import (figure_to_rgba_array,
                             rgb2gray_digital,
                             rgb2gray_human_eye,
                             simulate_colorblindness, 
                             rgb_to_lms,
                             lms_to_rgb,
                             get_palette)
diagfig.demo()
diagfig.demo("color_space")

#%%
rgb = []
pro = []
tri = []
for i in np.linspace(0,1, 20):
    rgb.append(get_palette(val = i, n = 20)[:, :, :, np.newaxis])
    pro.append(simulate_colorblindness(rgb[-1][:, :, :,0], "pro")[:, :, :, np.newaxis])
    tri.append(simulate_colorblindness(rgb[-1][:, :, :,0], "tri")[:, :, :, np.newaxis])
    # fig, ax = plt.subplots(FigureClass=diagfig_dev.FigureDiag, layout = "constrained")
    # ax.imshow(rgb)
    # ax.axis("off")
    # diaged_fig = fig.diag()
    # plt.close(fig)
def format_cube(data):
    rgbig = np.concatenate(data, axis = 3)
    shift = np.swapaxes((rgbig[:,:,:,:]*255).astype(np.uint8), 2,3)
    return shift

rgb = format_cube(rgb)
pro = format_cube(pro)
tri = format_cube(tri)
#%%
import plotly.express as px
import numpy as np
def show_cube(shift, name):
    fig = px.imshow(shift, animation_frame = 0)
    fig.write_html(f'{name}_cube_1.html', auto_open=True)
    fig = px.imshow(shift, animation_frame = 1)
    fig.write_html(f'{name}_cube_2.html', auto_open=True)
    fig = px.imshow(shift, animation_frame = 2)
    fig.write_html(f'{name}_cube_3.html', auto_open=True)

show_cube(rgb, "rgb")
show_cube(pro, "pro")
show_cube(tri, "tri")

#%%
x = np.arange(256, step = 10)

cube = np.swapaxes(np.array(np.meshgrid(x,x,x)), 0,3).astype(np.uint8)

fig = px.imshow(cube, animation_frame = 0)
fig.write_html('rgb_cube_1.html', auto_open=True)
fig = px.imshow(cube, animation_frame = 1)
fig.write_html('rgb_cube_2.html', auto_open=True)
fig = px.imshow(cube, animation_frame = 2)
fig.write_html('rgb_cube_3.html', auto_open=True)