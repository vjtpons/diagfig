# -*- coding: utf-8 -*-
"""
Example use of diagfig with plotly. [require plotly]
"""

import numpy as np

from diagfig import simulate_colorblindness
from diagfig.helpers import get_palette

#%% generate hsv colorcubes for protanopia and tritanopia
rgb = []
pro = []
tri = []
for i in np.linspace(0,1, 20):
    rgb.append(get_palette(val = i, n = 20)[:, :, :, np.newaxis])
    pro.append(simulate_colorblindness(rgb[-1][:, :, :,0], "pro")[:, :, :, np.newaxis])
    tri.append(simulate_colorblindness(rgb[-1][:, :, :,0], "tri")[:, :, :, np.newaxis])

def format_cube(data):
    rgbig = np.concatenate(data, axis = 3)
    shift = np.swapaxes((rgbig[:,:,:,:]*255).astype(np.uint8), 2,3)
    return shift

rgb = format_cube(rgb)
pro = format_cube(pro)
tri = format_cube(tri)
#%% visualisation of color cube with plotly
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
