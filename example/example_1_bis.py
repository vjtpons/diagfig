# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:08:33 2023

@author: vjtpons

This example is similar to example 1 and show how to use `diagfig` with an exported figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import diagfig

# creating a dummy figure in matplotlib
fig, ax = plt.subplots(layout = "constrained")
x = np.arange(10)
y_low = x * .5
y_high = x * 2
y = x.copy()
ax.fill_between(x, y_low, y_high, color = "tab:blue", alpha = .5)
ax.plot(x, y, color = "tab:red", lw = 2,ls = "--")

fig.savefig("example.png", dpi = 300)

diagfig.diagnose_figure("example.png")
