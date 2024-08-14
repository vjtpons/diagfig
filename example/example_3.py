# -*- coding: utf-8 -*-
"""
This example shows how to use the subclass for matplotlib `Figure` developped for diagfig.
"""

import numpy as np
import matplotlib.pyplot as plt
from diagfig import FigureDiag
import diagfig

# creating a dummy figure in matplotlib using the custom subclass of matplotlib.figure.Figure
fig, ax = plt.subplots(FigureClass=FigureDiag, layout = "constrained")
x = np.arange(10)
y_low = x * .5
y_high = x * 2
y = x.copy()
ax.fill_between(x, y_low, y_high, color = "tab:blue", alpha = .5)
ax.plot(x, y, color = "tab:red", lw = 2,ls = "--")

# diagnosing the figure using the default color config.
diaged_fig = fig.diag(config = diagfig.IXORA, dpi = 30)
