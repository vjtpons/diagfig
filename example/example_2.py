# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:35:44 2023

@author: vjtpons

This example shows how to use the decorator `@diag_it` for figure diagnosis.
"""

import numpy as np
import matplotlib.pyplot as plt
from diagfig import diag_it

# creating a dummy figure in matplotlib
@diag_it
def simple_plot():
    fig, ax = plt.subplots(layout = "constrained")
    x = np.arange(10)
    y_low = x * .5
    y_high = x * 2
    y = x.copy()
    ax.fill_between(x, y_low, y_high, color = "tab:blue", alpha = .5)
    ax.plot(x, y, color = "tab:red", lw = 2,ls = "--")
    return fig

# diagnosing the figure
simple_plot()