# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:01:53 2024

@author: vjtpons
"""
from abc import ABC, abstractmethod
import numpy as np


class ColorConfig(ABC):
    """
    Abstract base class for color configuration, providing methods for color space 
    transformations and color blindness simulations.

    This class defines static abstract methods that must be implemented by subclasses 
    to return specific conversion matrices and simulation matrices. This is implemented 
    because different studies recommend different conversion and transformation matrices.

    Methods
    -------
    rgb2lms() -> np.ndarray
        Returns the matrix for converting from RGB to LMS (Long, Medium, Short) 
        color space.
    lms2rgb() -> np.ndarray
        Returns the matrix for converting from LMS to RGB color space.
    protanopia(degree: float = 1.0) -> np.ndarray
        Returns the matrix for simulating protanopia (red-green color blindness) 
        with a specified severity.
    deuteranopia(degree: float = 1.0) -> np.ndarray
        Returns the matrix for simulating deuteranopia (red-green color blindness) 
        with a specified severity.
    tritanopia(degree: float = 1.0) -> np.ndarray
        Returns the matrix for simulating tritanopia (blue-yellow color blindness)
        with a specified severity.
    """
    @staticmethod
    @abstractmethod
    def rgb2lms() -> np.ndarray:
        """
        Return the matrix for converting from RGB to LMS (Long, Medium, Short) color 
        space.
        
        Returns
        -------
        numpy.ndarray
            A 3x3 matrix used for RGB to LMS conversion.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def lms2rgb() -> np.ndarray:
        """
        Return the matrix for converting from LMS to RGB color space.
        
        Returns
        -------
        numpy.ndarray
            A 3x3 matrix used for LMS to RGB conversion.
        """
        pass
    

    @staticmethod
    @abstractmethod
    def protanopia(degree: float = 1.0) -> np.ndarray:
        """
        Return the matrix for simulating protanopia (red-green color blindness) 
        with a specified severity.
        
        Parameters
        ----------
        degree : float, optional
            The severity of the protanopia simulation, where 1.0 represents full 
            severity, by default 1.0.
        
        Returns
        -------
        numpy.ndarray
            A 3x3 matrix used to simulate protanopia.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def deuteranopia(degree: float = 1.0) -> np.ndarray:
        """
        Return the matrix for simulating deuteranopia (red-green color blindness) 
        with a specified severity.
        
        Parameters
        ----------
        degree : float, optional
            The severity of the deuteranopia simulation, where 1.0 represents full 
            severity, by default 1.0.
        
        Returns
        -------
        numpy.ndarray
            A 3x3 matrix used to simulate deuteranopia.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def tritanopia(degree: float = 1.0) -> np.ndarray:
        """
        Return the matrix for simulating tritanopia (red-green color blindness) 
        with a specified severity.
        
        Parameters
        ----------
        degree : float, optional
            The severity of the tritanopia simulation, where 1.0 represents full 
            severity, by default 1.0.
        
        Returns
        -------
        numpy.ndarray
            A 3x3 matrix used to simulate tritanopia.
        """
        pass


class IXORA(ColorConfig):
    """
    Subclass of ColorConfig. the matrices implemented are taken from ixora.io. 
    They refer to Lindbloom's website for rgb to xyz conversion, and the Hunt–Pointer–Estevez 
    transformation matrix for xyz to lms conversion.
    They recalculated their own values for colourblindness simulation as detailed on their 
    website.
    
    References
    ----------
    
    Schmitz, J. (2016, August 28). Color Blindness Simulation Research. Ixora.Io. 
    https://ixora.io/projects/colorblindness/color-blindness-simulation-research/

    Lindbloom, B. (2017. April 06). RGB Working Space Information. Retrieved 14 August 2024, 
    from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html

    LMS color space. (2024). In Wikipedia. 
    https://en.wikipedia.org/w/index.php?title=LMS_color_space&oldid=1233434413#Hunt.2C_RLAB

    
    """
    @staticmethod
    def rgb2lms():
        return np.array([[0.31399022, 0.63951294, 0.04649755],
                         [0.15537241, 0.75789446, 0.08670142],
                         [0.01775239, 0.10944209, 0.87256922]])
    @staticmethod
    def lms2rgb():
        return np.array([[5.47221206, -4.6419601, 0.16963708],
                         [-1.1252419, 2.29317094, -0.1678952],
                         [0.02980165, -0.19318073, 1.16364789]])
    @staticmethod
    def protanopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1 - degree, 1.05118294 * degree, -0.05116099 * degree],
                         [0, 1, 0],
                         [0, 0, 1]])
    @staticmethod
    def deuteranopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1, 0, 0],
                         [0.9513092 * degree, 1 - degree, 0.04866992 * degree],
                         [0, 0, 1]])
    @staticmethod
    def tritanopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [-0.86744736 * degree, 1.86727089 * degree, 1 - degree]])


# Color setup taken from https://arxiv.org/pdf/1711.10662.pdf
class VIENOT(ColorConfig):
    """
    Subclass of ColorConfig. the matrices implemented are taken from Lee et al., (2017)
    which refers to the work of Vienot et al, (1999).
    Note that the tritanopia is taken from the Simulate-Correct-ColorBlindness code.
    
    References
    ----------
    
    Lee, J., & Santos, W. P. dos. (2011). An Adaptive Fuzzy-Based System to Simulate, 
    Quantify and Compensate Color Blindness. Integrated Computer-Aided Engineering, 
    18(1), 29–40. https://doi.org/10.3233/ICA-2011-0356
    
    Viénot, F., Brettel, H., & Mollon, J. D. (1999). Digital video colourmaps for 
    checking the legibility of displays by dichromats. Color Research & Application, 
    24(4), 243–252. https://doi.org/10.1002/(SICI)1520-6378(199908)24:4<243::AID-COL5>3.0.CO;2-3

    Thakkar, S. (2024). Tsarjak/Simulate-Correct-ColorBlindness [Python]. 
    https://github.com/tsarjak/Simulate-Correct-ColorBlindness (Original work published 2017)

    """
    @staticmethod
    def rgb2lms():
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]])

    @staticmethod
    def lms2rgb() -> np.ndarray:
        return np.array([[0.0809, -0.1305, 0.1167],
                         [-0.0102, 0.0540, -0.1136],
                         [-0.0004, -0.0041, 0.6935]])

    @staticmethod
    def protanopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1 - degree, 2.02344 * degree, -2.52581 * degree],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def deuteranopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1, 0, 0],
                         [0.494207 * degree, 1 - degree, 1.24827 * degree],
                         [0, 0, 1]])

    @staticmethod
    def tritanopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [-0.395913 * degree, 0.801109 * degree, 1 - degree]])


class RUMINSKI(ColorConfig):
    """
    Subclass of ColorConfig. the matrices implemented are taken from Ruminski et al., (2012)
    
    References
    ----------
    
    Ruminski, J., Bajorek, M., Ruminska, J., Wtorek, J., & Bujnowski, A. (2012). 
    Computerized Color Processing for Dichromats. In Z. S. Hippe, J. L. Kulikowski, & 
    T. Mroczek (Eds.), Human – Computer Systems Interaction: Backgrounds and 
    Applications 2: Part 1 (pp. 453–470). Springer. https://doi.org/10.1007/978-3-642-23187-2_29
    """
    @staticmethod
    def rgb2lms():
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]])

    @staticmethod
    def lms2rgb() -> np.ndarray:
        return np.array([[0.0809, -0.1305, 0.1167],
                         [-0.0102, 0.0540, -0.1136],
                         [-0.0004, -0.0041, 0.6935]])

    @staticmethod
    def protanopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1 - degree, 2.02344 * degree, -2.52581 * degree],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def deuteranopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1, 0, 0],
                         [0.494207 * degree, 1 - degree, 1.24827 * degree],
                         [0, 0, 1]])

    @staticmethod
    def tritanopia(degree: float = 1.0) -> np.ndarray:
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [-0.012245 * degree, 0.0720345 * degree, 1 - degree]])
