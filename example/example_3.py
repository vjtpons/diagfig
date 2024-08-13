# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 10:59:06 2023

@author: vjtpons

This example shows how to use the subclass for matplotlib `Figure` developped for diagfig.
"""

import numpy as np
import matplotlib.pyplot as plt
from diagfig import FigureDiag

# creating a dummy figure in matplotlib using the custom subclass of matplotli.figure.Figure
fig, ax = plt.subplots(FigureClass=FigureDiag, layout = "constrained")
x = np.arange(10)
y_low = x * .5
y_high = x * 2
y = x.copy()
ax.fill_between(x, y_low, y_high, color = "tab:blue", alpha = .5)
ax.plot(x, y, color = "tab:red", lw = 2,ls = "--")

# diagnosing the figure
diaged_fig = fig.diag()
