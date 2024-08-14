# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:51:08 2023

@author: vjtpons
"""
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib as mpl
from diagfig.colour_config import ColorConfig, IXORA, VIENOT, RUMINSKI
#%%
def figure_to_rgba_array(fig: mpl.figure.Figure, draw: bool = True) -> npt.NDArray:
    """
    Convert a matplotlib figure to an RGBA numpy array.
    
    Parameters
    ----------
    fig : mpl.figure.Figure
        The figure to convert to an RGBA numpy array.
    draw : bool, optional
        Whether to draw the figure canvas before converting, by default True.
    
    Returns
    -------
    np.ndarray
        The RGBA numpy array representing the figure.
    
    Examples
    --------
    
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.arange(10))
    >>> rgba_fig = figure_to_rgba_array(fig)
    """
    if draw:
        fig.canvas.draw()

    rgba_buffer = fig.canvas.buffer_rgba()
    (width, height) = fig.canvas.get_width_height()
    rgba_array = np.frombuffer(rgba_buffer, dtype=np.uint8).reshape((height, width, 4))
    return rgba_array / 255


def rgb2gray_human_eye(rgb: npt.NDArray) -> npt.NDArray:
    """
    Convert an RGB array to grayscale using the human eye perception weights.
    
    Parameters
    ----------
    rgb : np.ndarray
        A 3-dimensional numpy array representing an RGB image.
        
    Returns
    -------
    np.ndarray
        A 2-dimensional numpy array representing the grayscale image.
        
    Examples
    --------
    >>> from PIL import Image
    >>> import numpy as np
    >>> from diagfig.helpers import rgb2gray_human_eye
    >>> 
    >>> # Load an example image
    >>> img = np.asarray(Image.open("example.png"))
    >>> 
    >>> # Convert the image to grayscale
    >>> gray_img = rgb2gray_human_eye(img)
    """
    # Apply the human eye perception weights to the RGB channels
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    return gray


def rgb2gray_digital(rgb: npt.NDArray) -> npt.NDArray:
    """
    Take a rgb array as argument and return an array adjusted to digital vision in 
    black and white.
    
    Parameters
    ----------
    rgb : np.ndarray
        A numpy array representing an RGB image.
    
    Returns
    -------
    np.ndarray
        A grayscale numpy array.
    
    Examples
    --------
    
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.arange(10))
    >>> rgba_fig = plt2arr(fig)
    >>> fig_gray_digital = rgb2gray_digital(rgba_fig)
    """
    gray = np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])
    return gray


def rgb_to_lms(rgb: npt.NDArray, config: ColorConfig = IXORA) -> npt.NDArray:
    """
    Convert an RGB image array to an LMS (Long, Medium, Short) color space array.

    Parameters
    ----------
    rgb : np.ndarray
        A numpy array of shape (height, width, 3) representing the RGB color channels.
    config : ColorConfig, optional
        Configuration object providing the necessary color transformation matrices:
            - RGB to LMS conversion matrix,
        by default IXORA

    Returns
    -------
    np.ndarray
        A numpy array of shape (height, width, 3) representing the LMS color channels.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.arange(10))
    >>> rgba_fig = plt2arr(fig)
    >>> fig_lms = rgb_to_lms(rgba_fig)
    """
    lms_matrix = config.rgb2lms()
    lms = np.tensordot(rgb, lms_matrix, axes=([2], [1]))
    return lms


def lms_to_rgb(lms: npt.NDArray, config: ColorConfig = IXORA) -> npt.NDArray:
    """
    Convert an LMS (Long, Medium, Short) color space array back to an RGB array.

    Parameters
    ----------
    lms : np.ndarray
        A numpy array of shape (height, width, 3) representing the LMS color channels.
    config : ColorConfig, optional
        Configuration object providing the necessary color transformation matrices:
            - LMS to RGB conversion matrix,
        by default IXORA.

    Returns
    -------
    np.ndarray
        A numpy array of shape (height, width, 3) representing the RGB color channels.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.arange(10))
    >>> rgba_fig = plt2arr(fig)
    >>> fig_lms = rgb_to_lms(rgba_fig)
    >>> fig_rgb = lms_to_rgb(fig_lms)
    """
    rgb_matrix = config.lms2rgb()
    rgb = np.tensordot(lms, rgb_matrix, axes=([2], [1]))
    return rgb


def simulate_colorblindness(rgb: npt.NDArray, colorblind_type: str,
                            config: ColorConfig = IXORA) -> npt.NDArray:
    """
    Simulate the effects of colorblindness on an RGB image by adjusting it according 
    to a specific type of colorblindness. The transformation is based on a simulation 
    matrix that is specific to the type of colorblindness.
    
    Parameters
    ----------
    rgb : np.ndarray
        A numpy array representing an RGB image, with shape (height, width, 3).
    colorblind_type : str
        The type of colorblindness to simulate. Can be one of the following:
        - 'protanopia', 'p', 'pro' for protanopia
        - 'deuteranopia', 'd', 'deut' for deuteranopia
        - 'tritanopia', 't', 'tri' for tritanopia
    config : ColorConfig, optional
        Configuration object providing the necessary color transformation matrices:
            - RGB to LMS conversion matrix,
            - LMS to RGB conversion matrix,
            - protanopia conversion matrix,
            - deuteranopia conversion matrix,
            - tritanopia conversion matrix,
        by default IXORA.
    
    Returns
    -------
    np.ndarray
        A numpy array representing the simulated RGB image, with shape (height, width, 3).
    
    Raises
    ------
    ValueError
        If `colorblind_type` is not recognized.
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.arange(10))
    >>> rgba_fig = plt2arr(fig)
    >>> fig_colorblind = simulate_colorblindness(rgba_fig[:,:,:-1], 'd')
    """
    # convert RGB image to LMS color space
    lms_img = rgb_to_lms(rgb, config = config)

    # choose simulation matrix based on colorblind type
    if colorblind_type.lower() in ['protanopia', 'p', 'pro']:
        cvd_sim_matrix = config.protanopia()
    elif colorblind_type.lower() in ['deuteranopia', 'd', 'deut']:
        cvd_sim_matrix = config.deuteranopia()
    elif colorblind_type.lower() in ['tritanopia', 't', 'tri']:
        cvd_sim_matrix = config.tritanopia()
    else:
        raise ValueError(f"{colorblind_type} is an unrecognized colorblindness type.")

    # apply simulation matrix to LMS image
    lms_img = np.tensordot(lms_img, cvd_sim_matrix, axes=([2], [1]))

    # convert back to RGB color space
    rgb_img = lms_to_rgb(lms_img, config = config)
    rgb_img = np.clip(rgb_img, 0.0, 1.0)
    return rgb_img


def hsv_to_rgb_vec(hsv):
    """
    Convert an HSV image array to an RGB image array using a vectorized approach.
    
    This function performs a vectorized conversion of an HSV (Hue, Saturation, Value) 
    image to an RGB (Red, Green, Blue) image. The method is based on the algorithm 
    used in the standard `colorsys` module optimized to handle large arrays.
    See references for original code.

    Parameters
    ----------
    hsv : np.ndarray
        A numpy array of shape (..., 3), where the last dimension represents the HSV 
        color channels.

    Returns
    -------
    np.ndarray
        A numpy array of shape (..., 3) where the last dimension represents the RGB 
        color channels.

    Examples
    --------
    >>> hsv_image = np.array([[[0.5, 0.5, 0.5], [0.7, 0.8, 0.9]]])
    >>> rgb_image = hsv_to_rgb_vec(hsv_image)
    
    References
    ----------
    Numpy RGB to HSV. Gist. Retrieved 14 August 2024, 
    from https://gist.github.com/PolarNick239/691387158ff1c41ad73c

    """
    input_shape = hsv.shape
    hsv = hsv.reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    i = np.int32(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    rgb = np.zeros_like(hsv)
    v, t, p, q = v.reshape(-1, 1), t.reshape(-1, 1), p.reshape(-1, 1), q.reshape(-1, 1)
    rgb[i == 0] = np.hstack([v, t, p])[i == 0]
    rgb[i == 1] = np.hstack([q, v, p])[i == 1]
    rgb[i == 2] = np.hstack([p, v, t])[i == 2]
    rgb[i == 3] = np.hstack([p, q, v])[i == 3]
    rgb[i == 4] = np.hstack([t, p, v])[i == 4]
    rgb[i == 5] = np.hstack([v, p, q])[i == 5]
    rgb[s == 0.0] = np.hstack([v, v, v])[s == 0.0]
    return rgb.reshape(input_shape)


def get_palette(val: float = 1., n: int = 100, show: bool = False):
    """
    Generate and optionally display an HSV color palette in RGB format.

    Parameters
    ----------
    val : float, optional
        The value (brightness) component of the HSV color space, by default 1.0.
    n : int, optional
        The size of the palette, defining both the width and height of the generated 
        image grid, by default 100.
    show : bool, optional
        If True, displays the generated palette using matplotlib, by default False.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n, n, 3) representing the RGB color channels of the 
        generated palette.

    Examples
    --------
    >>> palette = get_palette(val=0.8, n=200, show=True)
    >>> print(palette.shape)
    (200, 200, 3)
    """
    xx = np.linspace(0,1,n)
    yy = np.linspace(0,1,n)
    X,Y = np.meshgrid(xx, yy)
    
    h = np.flip((X - X.min()) / (X.max() - X.min()))
    s = np.flip((Y - Y.min()) / (Y.max() - Y.min()))
    v = np.ones_like(s) * val
    
    hsv = np.concatenate((h[:, :, np.newaxis],
                          s[:, :, np.newaxis],
                          v[:, :, np.newaxis]), axis = 2)
    rgb = hsv_to_rgb_vec(hsv)
    if show:
        fig, ax = plt.subplots(layout = "constrained")
        ax.imshow(rgb, aspect = "auto")
        ax.axis("off")
    return rgb

