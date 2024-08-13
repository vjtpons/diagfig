# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:01:53 2024

@author: vjtpons
"""
from abc import ABC, abstractmethod
import numpy as np


class ColorConfig(ABC):

    @staticmethod
    @abstractmethod
    def rgb2lms() -> np.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def lms2rgb() -> np.ndarray:
        pass
    

    @staticmethod
    @abstractmethod
    def protanopia(degree: float = 1.0) -> np.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def deuteranopia(degree: float = 1.0) -> np.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def tritanopia(degree: float = 1.0) -> np.ndarray:
        pass


class IXORA(ColorConfig):
    """Color setup taken from ixora.io"""
    @staticmethod
    def rgb2lms():
        """Conversion matrix from rgb to lms taken from ixora.io"""
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
    Taken from: Digital Video Colourmaps for Checking the Legibility of 
    Displays by Dichromats, Francoise Vienot, Hans Brettel, and, John D. Mollon,
    1999.
    """
    @staticmethod
    def rgb2lms():
        """Used in Vienot et al."""
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]])

    @staticmethod
    def lms2rgb() -> np.ndarray:
        """Used in Vienot et al."""
        return np.array([[0.0809, -0.1305, 0.1167],
                         [-0.0102, 0.0540, -0.1136],
                         [-0.0004, -0.0041, 0.6935]])

    @staticmethod
    def protanopia(degree: float = 1.0) -> np.ndarray:
        """Modified from Vienot et al."""
        return np.array([[1 - degree, 2.02344 * degree, -2.52581 * degree],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def deuteranopia(degree: float = 1.0) -> np.ndarray:
        """Modified from Vienot et al."""
        return np.array([[1, 0, 0],
                         [0.494207 * degree, 1 - degree, 1.24827 * degree],
                         [0, 0, 1]])

    @staticmethod
    def tritanopia(degree: float = 1.0) -> np.ndarray:
        """Not used in Vienot et al."""
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [-0.395913 * degree, 0.801109 * degree, 1 - degree]])

class RUMINSKI(ColorConfig):
    """
    doi: 978-3-642-23187-2
    """
    @staticmethod
    def rgb2lms():
        """Used in Vienot et al."""
        return np.array([[17.8824, 43.5161, 4.11935],
                         [3.45565, 27.1554, 3.86714],
                         [0.0299566, 0.184309, 1.46709]])

    @staticmethod
    def lms2rgb() -> np.ndarray:
        """Used in Vienot et al."""
        return np.array([[0.0809, -0.1305, 0.1167],
                         [-0.0102, 0.0540, -0.1136],
                         [-0.0004, -0.0041, 0.6935]])

    @staticmethod
    def protanopia(degree: float = 1.0) -> np.ndarray:
        """Modified from Vienot et al."""
        return np.array([[1 - degree, 2.02344 * degree, -2.52581 * degree],
                         [0, 1, 0],
                         [0, 0, 1]])

    @staticmethod
    def deuteranopia(degree: float = 1.0) -> np.ndarray:
        """Modified from Vienot et al."""
        return np.array([[1, 0, 0],
                         [0.494207 * degree, 1 - degree, 1.24827 * degree],
                         [0, 0, 1]])

    @staticmethod
    def tritanopia(degree: float = 1.0) -> np.ndarray:
        """Not used in Vienot et al."""
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [-0.012245 * degree, 0.0720345 * degree, 1 - degree]])



