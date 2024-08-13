# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:51:08 2023

@author: vjtpons
"""

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib as mpl
#%% conversion matrix
RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])

XYZ2LMS = np.array([[ 0.4002, 0.7076, -0.0808],
                    [-0.2263, 1.1653,  0.0457],
                    [      0,      0,  0.9182]])

RGB2LMS = XYZ2LMS @ RGB2XYZ

RGB2LMS = np.array([[0.31399022, 0.63951294, 0.04649755],
                    [0.15537241, 0.75789446, 0.08670142],
                    [0.01775239, 0.10944209, 0.87256922]])

LMS2RGB = np.linalg.inv(RGB2LMS)

LMS2RGB = np.array([[5.47221206, -4.6419601, 0.16963708],
                    [-1.1252419, 2.29317094, -0.1678952],
                    [0.02980165, -0.19318073, 1.16364789]])

cones_to_rgb = np.array([
            # L        M        S
            [4.97068857, -4.14354132, 0.17285275],  # R
            [-0.90913894, 2.15671326, -0.24757432],  # G
            [-0.03976551, -0.14253782, 1.18230333]]) 
rgb_to_cones = np.linalg.inv(cones_to_rgb)
#%%
def figure_to_rgba_array(fig: mpl.figure.Figure, draw: bool = True) -> npt.NDArray:
    """
    Convert a matplotlib figure to an RGBA numpy array.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert to an RGBA numpy array.
    draw : bool, optional
        Whether to draw the figure canvas before converting, by default True.
    
    Returns
    -------
    numpy.ndarray
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
    rgb : numpy.ndarray
        A 3-dimensional numpy array representing an RGB image.
        
    Returns
    -------
    numpy.ndarray
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
    Take a rgb array as argument and return an array adjusted to digital vision in black and white.
    
    Parameters
    ----------
    rgb : numpy.ndarray
        A numpy array representing an RGB image.
    
    Returns
    -------
    numpy.ndarray
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

def rgb_to_lms(rgb: npt.NDArray, cm = 1) -> npt.NDArray:
    """
    Convert from RGB array to LMS array.
    
    Parameters
    ----------
    rgb : numpy.ndarray
        A numpy array of shape (height, width, 3) where the last dimension represents RGB color channels.
        
    Returns
    -------
    numpy.ndarray
        A numpy array of shape (height, width, 3) where the last dimension represents LMS color channels.
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.arange(10))
    >>> rgba_fig = plt2arr(fig)
    >>> fig_lms = rgb_to_lms(rgba_fig)
    """
    # if cm = 1:
    # lms_matrix = np.array(
    #     [[0.3904725 , 0.54990437, 0.00890159],
    #     [0.07092586, 0.96310739, 0.00135809],
    #     [0.02314268, 0.12801221, 0.93605194]]
    # )
    lms_matrix = RGB2LMS
    # elic cm = 2:
        
    lms = np.tensordot(rgb, lms_matrix, axes=([2], [1]))
    # lms = np.dot(lms_matrix, np.transpose(rgb))
    # lms[lms<0] = 0
    # lms[lms>255] = 255
    return lms

def lms_to_rgb(lms: npt.NDArray) -> npt.NDArray:
    """
    Convert from LMS array to RGB array.
    
    Parameters
    ----------
    lms : numpy.ndarray
        A numpy array of shape (height, width, 3) where the last dimension represents LMS color channels.
    
    Returns
    -------
    numpy.ndarray
        A numpy array of shape (height, width, 3) where the last dimension represents RGB color channels.
    
    Examples
    --------
    
    >>> fig, ax = plt.subplots()
    >>> ax.plot(np.arange(10))
    >>> rgba_fig = plt2arr(fig)
    >>> fig_lms = rgb_to_lms(rgba_fig)
    >>> fig_rgb = lms_to_rgb(fig_lms)
    """
    # rgb_matrix = np.array(
    #     [[ 2.85831110e+00, -1.62870796e+00, -2.48186967e-02],
    #     [-2.10434776e-01,  1.15841493e+00,  3.20463334e-04],
    #     [-4.18895045e-02, -1.18154333e-01,  1.06888657e+00]]
    # )
    rgb_matrix = LMS2RGB
    rgb = np.tensordot(lms, rgb_matrix, axes=([2], [1]))
    # rgb = np.dot(rgb_matrix, np.transpose(lms))
    # rgb[rgb<0] = 0
    # rgb[rgb>255] = 255
    return rgb

def simulate_colorblindness(rgb: npt.NDArray, colorblind_type: str) -> npt.NDArray:
    """
    Simulate colorblindness by transforming an RGB image into a new RGB image that is adjusted for the specified
    type of colorblindness. The transformation is based on a simulation matrix that is specific to the type of
    colorblindness.

    Parameters
    ----------
    rgb : npt.NDArray
        An array representing an RGB image, with shape (height, width, 3).
    colorblind_type : str
        A string indicating the type of colorblindness to simulate. Can be 'protanopia', 'p', 'pro' for protanopia,
        'deuteranopia', 'd', 'deut' for deuteranopia, or 'tritanopia', 't', 'tri' for tritanopia.

    Returns
    -------
    npt.NDArray
        An array representing the simulated RGB image, with shape (height, width, 3).

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
    lms_img = rgb_to_lms(rgb)

    # choose simulation matrix based on colorblind type
    if colorblind_type.lower() in ['protanopia', 'p', 'pro']:
        cvd_sim_matrix = np.array([[0, 1.05118294, -0.05116099], [0, 1, 0], [0, 0, 1]],
        # cvd_sim_matrix = np.array([[0, 0.90822864, 0.008192], [0, 1, 0], [0, 0, 1]],
                                  dtype=np.float16)
    elif colorblind_type.lower() in ['deuteranopia', 'd', 'deut']:
        cvd_sim_matrix = np.array([[1, 0, 0], [0.9513092, 0, 0.04866992], [0, 0, 1]],
        # cvd_sim_matrix = np.array([[1, 0, 0], [1.10104433, 0, -0.00901975], [0, 0, 1]],
                                  dtype=np.float16)
    elif colorblind_type.lower() in ['tritanopia', 't', 'tri']:
        cvd_sim_matrix = np.array([[1, 0, 0], [0, 1, 0], [-0.86744736, 1.86727089, 0]],
        # cvd_sim_matrix = np.array([[1, 0, 0], [0, 1, 0], [-0.15773032, 1.19465634, 0]],
                                  dtype=np.float16)
    else:
        raise ValueError(f"{colorblind_type} is an unrecognized colorblindness type.")

    # apply simulation matrix to LMS image
    lms_img = np.tensordot(lms_img, cvd_sim_matrix, axes=([2], [1]))

    # convert back to RGB color space
    rgb_img = lms_to_rgb(lms_img)
    rgb_img = np.clip(rgb_img, 0.0, 1.0)
    return rgb_img#.astype(np.uint8)

def hsv_to_rgb_vec(hsv):
    """
    Vectorized conversion from hsv to rgb.
    Code taken online to not increase number of dependencies.
    It is a vectorized version of the colosys code.
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
    """Show a color palette."""
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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    