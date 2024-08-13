# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:04:28 2023

@author: vjtpons
"""

from typing import Callable
import importlib.resources
from pathlib import Path

from PIL import Image
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from diagfig.utils import (figure_to_rgba_array,
                           rgb2gray_digital,
                           rgb2gray_human_eye,
                           simulate_colorblindness)

from diagfig.colour_config import ColorConfig, IXORA, VIENOT, RUMINSKI


def diagnose_figure(figure, config: ColorConfig = IXORA, aspect = None) -> mpl.figure.Figure:
    """
    Diagnose the given figure for color blindness and display it in various color spaces.
    
    Parameters
    ----------
    figure : mpl.figure.Figure or npt.NDArray or str
        The input figure to be diagnosed for color blindness. 
        It can be a matplotlib figure, a 3 or 4 dimensional numpy array, 
        or a path or filename that can be opened with Pillow.
        
    Returns
    -------
    mpl.figure.Figure
        The diagnosed figure in various color spaces.

    Raises
    ------
    ValueError
        If `figure` is a file name and can't be open with `PIL`.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> fig_diag = diagnose_figure(fig)
    """
    if isinstance(figure, mpl.figure.Figure):
        fig_rgba = figure_to_rgba_array(figure)
    elif isinstance(figure, np.ndarray):
        if figure.ndim in [3,4]:
            fig_rgba = figure.copy()
    elif isinstance(figure, str):
        try:
            fig_rgba = np.asarray(Image.open(figure)) / 255
        except: 
            raise ValueError(f"Autoload of {figure} not supported. Load the figure as a matplotlib figure or a numpy array.")
    elif isinstance(figure, Path):
        fig_rgba = np.asarray(Image.open(figure)) / 255
    # Convert the input figure to grayscale and simulate color blindness in various color spaces.
    fig_gray_human = rgb2gray_human_eye(fig_rgba)
    fig_gray_digital = rgb2gray_digital(fig_rgba)
    fig_protanopia = simulate_colorblindness(fig_rgba[:,:,:-1], "p", config = config)
    fig_deuteranopia = simulate_colorblindness(fig_rgba[:,:,:-1], "d", config = config)
    fig_tritanopia = simulate_colorblindness(fig_rgba[:,:,:-1], "t", config = config)
    
    # Create a 3x2 subplot grid to display the original and diagnosed figures.
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 9), layout="constrained",
                           sharex=True, sharey=True)
    
    # Display the original figure, grayscale figure, and diagnosed figures in various color spaces.
    ax[0,0].imshow(fig_rgba, aspect=aspect)
    ax[0,1].imshow(fig_gray_human, cmap="gray", aspect=aspect)
    ax[0,2].imshow(fig_gray_digital, cmap="gray", aspect=aspect)
    ax[1,0].imshow(fig_protanopia, aspect=aspect)
    ax[1,1].imshow(fig_deuteranopia, aspect=aspect)
    ax[1,2].imshow(fig_tritanopia, aspect=aspect)
    
    # Add titles to the subplots and turn off the axes ticks and labels.
    title_list = ["Original", "ITU-R BT.601 luma weights", "ITU-R Recommendation BT.709", "Protanopia", "Deuteranopia", "Tritanopia"]
    for i, axi in enumerate(ax.flatten()):
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_title(title_list[i])
        axi.axis('off')

    return fig


def diag_it(func: Callable[..., plt.Figure]) -> Callable[..., None]:
    """
    Decorator for plotting functions. Decorates functions that return a `matplotlib.pyplot` Figure object.
    The decorated function will produce the original figure and several diagnostic versions of it.

    Parameters
    ----------
    func : callable
        The function to decorate. Must return a `matplotlib.pyplot` Figure object.

    Returns
    -------
    callable
        The decorated function.

    Examples
    --------
    >>> @diag_it
    ... def my_plot():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3], [4, 5, 6])
    ...     return fig
    ...
    >>> my_plot()
    """
    def wrapper(*args, **kwargs) -> None:
        fig = func(*args, **kwargs)
        diagnose_figure(fig)
    return wrapper


class FigureDiag(Figure):
    """
    A `matplotlib.pyplot` Figure subclass with a diagnosis function.
    The `diag` method produces the original figure and several colour diagnostic versions of it.

    Examples
    --------
    >>> fig = FigureDiag()
    >>> ax = fig.add_subplot()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> fig.diag()
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def diag(self, config: ColorConfig = IXORA, aspect = None) -> None:
        """
        Produces the original figure and several diagnostic versions of it.
        """
        diagnose_figure(self, config=config, aspect=aspect)

resources_path = importlib.resources.files("diagfig.data")
list_resources = { path.name.rsplit(".")[0]: path for path in resources_path.iterdir()}
def demo(key: str = None, config = IXORA):
    if key == "color_space":
        diagnose_figure(list_resources["color_space"], config=config, aspect="auto")
    else:
        diagnose_figure(list_resources["example"], config=config)