# -*- coding: utf-8 -*-
"""
This example shows how to use the diagnose_figure feature of diagfig.
"""
import numpy as np
import matplotlib.pyplot as plt
import diagfig

# creating a dummy figure in matplotlib
fig, ax = plt.subplots(layout = "constrained", figsize = (6,6), dpi = 100)
x = np.arange(10)
y_low = x * .5
y_high = x * 2
y = x.copy()
ax.fill_between(x, y_low, y_high, color = "tab:blue", alpha = .5)
ax.plot(x, y, color = "tab:red", lw = 2,ls = "--")

#%% diagnosing the figure with different color configuration, or quality
fig_diag = diagfig.diagnose_figure(fig, config=diagfig.IXORA, aspect="auto", dpi=30)
fig_diag = diagfig.diagnose_figure(fig, config=diagfig.VIENOT, aspect="auto")
fig_diag = diagfig.diagnose_figure(fig, config=diagfig.RUMINSKI, aspect="auto")
