# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:04:28 2023

@author: vjtpons
"""

from typing import Callable, Optional
import warnings
import importlib.resources
from pathlib import Path

from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from diagfig.utils import (figure_to_rgba_array,
                           rgb2gray_digital,
                           rgb2gray_human_eye,
                           simulate_colorblindness)

from diagfig.colour_config import ColorConfig, IXORA


resources_path = importlib.resources.files("diagfig.data") # import resources
dict_resources = { path.name.rsplit(".")[0]: path for path in resources_path.iterdir()} # resources dict


def diagnose_figure(figure: Figure | np.ndarray | Path | str,
                    config: ColorConfig = IXORA,
                    aspect: Optional[int] = None,
                    dpi: int = None) -> mpl.figure.Figure:
    """
    Diagnose a given figure for color blindness by displaying it in various color spaces, 
    including grayscale and simulated color blindness conditions.

    Parameters
    ----------
    figure : mpl.figure.Figure | np.ndarray | Path | str
        The input figure to be diagnosed. This can be a matplotlib Figure object, 
        a numpy rgba array, or a file path to an image that can be opened with Pillow.
    config : ColorConfig, optional
        Configuration for color blindness simulation, by default IXORA.
    aspect : Optional[int], optional
        Aspect ratio for displaying images, by default None.
    dpi : int, optional
        Change the dpi of the fiture to test the effect of low resolution, by default None.
        Only usable if figure is a matplotlib figure.

    Returns
    -------
    mpl.figure.Figure
        A matplotlib Figure containing the original and diagnosed images.

    Raises
    ------
    ValueError
        If `figure` is a file path that cannot be opened with Pillow.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> fig_diag = diagnose_figure(fig)
    """
    if dpi:
        if isinstance(figure, mpl.figure.Figure):
            print("hey")
            figure.set_dpi(dpi)
        else:
            warnings.warn("Modifying dpi is only supported for matplotlib.figure.Figure object")
    if isinstance(figure, mpl.figure.Figure):
        fig_rgba = figure_to_rgba_array(figure)
    elif isinstance(figure, np.ndarray):
        if figure.ndim in [3,4]:
            fig_rgba = figure.copy()
    elif isinstance(figure, str):
        try:
            fig_rgba = np.asarray(Image.open(figure)) / 255
        except: 
            raise ValueError(f"""Autoload of {figure} not supported. Load the figure as 
                                 a matplotlib figure or a numpy array.""")
    elif isinstance(figure, Path):
        fig_rgba = np.asarray(Image.open(figure)) / 255
    # Convert the input figure to grayscale and simulate color blindness in various color spaces.
    fig_gray_human = rgb2gray_human_eye(fig_rgba)
    fig_gray_digital = rgb2gray_digital(fig_rgba)
    fig_protanopia = simulate_colorblindness(fig_rgba[:,:,:3], "p", config = config)
    fig_deuteranopia = simulate_colorblindness(fig_rgba[:,:,:3], "d", config = config)
    fig_tritanopia = simulate_colorblindness(fig_rgba[:,:,:3], "t", config = config)
    
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
    title_list = ["Original", "ITU-R BT.601 luma weights", "ITU-R Recommendation BT.709",
                  "Protanopia", "Deuteranopia", "Tritanopia"]
    for i, axi in enumerate(ax.flatten()):
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_title(title_list[i])
        axi.axis('off')
    return fig


def diag_it(config: ColorConfig = IXORA, aspect: Optional[int] = None, dpi: int = None):
    """
    Decorator for matplotlib plotting functions, generating diagnostic visualizations for 
    color blindness.

    Parameters
    ----------
    config : ColorConfig, optional
        Configuration settings for simulating color blindness, by default IXORA.
    aspect : Optional[int], optional
        Aspect ratio for displaying images, by default None.
    dpi : int, optional
        Change the dpi of the fiture to test the effect of low resolution, by default None.

    Returns
    -------
    Callable
        A decorator that modifies a function to generate and display color blindness 
        diagnostics for the figure it returns.

    Examples
    --------
    >>> @diag_it(dpi = 100)
    ... def my_plot():
    ...     fig, ax = plt.subplots()
    ...     ax.plot([1, 2, 3], [4, 5, 6])
    ...     return fig
    ...
    >>> my_plot()
    """
    def diag_inner(func: Callable[..., plt.Figure]) -> Callable[..., None]:
        def wrapper(*args, **kwargs) -> None:
            fig = func(*args, **kwargs)
            diagnose_figure(fig, config=config, aspect=aspect, dpi=dpi)
        return wrapper
    return diag_inner


class FigureDiag(Figure):
    """
    A `matplotlib.pyplot` Figure subclass with a diagnosis function.
    The `diag` method produces the original figure and several colour diagnostic 
    versions of it.

    Examples
    --------
    >>> fig = FigureDiag()
    >>> ax = fig.add_subplot()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> fig.diag()
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def diag(self, config: ColorConfig = IXORA, aspect = None, dpi: int = None) -> None:
        """
        Generate and display the original figure alongside diagnostic versions simulating 
        different types of color blindness.

        Parameters
        ----------
        config : ColorConfig, optional
            Configuration for simulating color blindness, by default IXORA.
        aspect : Optional[int], optional
            Aspect ratio for displaying images, by default None.
        dpi : int, optional
            Change the dpi of the fiture to test the effect of low resolution, 
            by default None.
        """
        diagnose_figure(self, config=config, aspect=aspect, dpi=dpi)


def demo(key: Optional[str] = None, config = IXORA):
    """
    Demonstrates the color blindness diagnosis on a predefined figure resource.
    
    Parameters
    ----------
    key : Optional[str], optional
        Key to select a specific figure for demonstration. If "color_space", a figure 
        illustrating color spaces will be used; otherwise, a default example figure 
        is used. By default None.
    config : ColorConfig, optional
        Configuration for simulating color blindness, by default IXORA.
    
    Examples
    --------
    >>> demo(key="color_space")
    >>> demo()
    """
    if key == "color_space":
        diagnose_figure(dict_resources["color_space"], config=config, aspect="auto")
    else:
        diagnose_figure(dict_resources["example"], config=config)
