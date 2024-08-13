# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 13:40:44 2023

@author: vjtpons
"""

import numpy as np
import matplotlib.pyplot as plt
import diagfig
import unittest
from numpy.testing import (assert_equal, assert_array_almost_equal)

# diagfig.helpers.figure_to_rgba_array
# diagfig.helpers.rgb2gray_human_eye
# diagfig.helpers.rgb2gray_digital
# diagfig.helpers.rgb_to_lms
# diagfig.helpers.lms_to_rgb
# diagfig.helpers.simulate_colorblindness

class TestTransFHelpers(unittest.TestCase):
    def setUp(self):
        self.mat4test = np.array([[[0.10483567, 0.41914279, 0.94802049],
                                   [0.37972101, 0.55484962, 0.64466938],
                                   [0.32384113, 0.82543779, 0.04458672]],
                                  [[0.79287661, 0.33183057, 0.02537918],
                                   [0.83539374, 0.63318922, 0.87633125],
                                   [0.47050339, 0.76981426, 0.61729444]],
                                  [[0.87730152, 0.83465613, 0.54251742],
                                   [0.84584869, 0.60231976, 0.52114862],
                                   [0.84066896, 0.04366148, 0.92686388]]])
        self.expected_hu_eye = np.array([[0.38544654, 0.51268765, 0.58641098],
                                     [0.43466859, 0.72128302, 0.662886  ],
                                     [0.81401556, 0.66579682, 0.38256772]])
        self.expected_dig = np.array([[0.39050607, 0.52410226, 0.66242089],
                                      [0.40772317, 0.69373276, 0.69516884],
                                      [0.82263013, 0.64823345, 0.27687248]])
        self.expected_rgb_to = np.array([[[0.27986279, 0.41240258, 0.94347799],
                                          [0.45912343, 0.56218733, 0.68325931],
                                          [0.5807598 , 0.8180145 , 0.15489615]],
                                         [[0.49229751, 0.3758584 , 0.08458388],
                                          [0.68219254, 0.67027037, 0.92068077],
                                          [0.61253776, 0.775623  , 0.68725399]],
                                         [[0.80637244, 0.86682364, 0.63497377],
                                          [0.66613797, 0.64079892, 0.58450167],
                                          [0.36051831, 0.10293463, 0.89263727]]])
        self.expected_lms_to = np.array([[[-0.40653687,  0.463784  ,  0.95941132],
                                          [ 0.16567293,  0.56304617,  0.60761423],
                                          [-0.41986499,  0.88806631, -0.06343645]],
                                         [[ 1.72520305,  0.21755681, -0.04529296],
                                          [ 1.33478548,  0.55798078,  0.82689042],
                                          [ 0.07572201,  0.79295188,  0.54915169]],
                                         [[ 1.13472501,  0.78243723,  0.44452162],
                                          [ 1.42376148,  0.51990723,  0.45044989],
                                          [ 2.30877807, -0.12603085,  0.95033835]]])
        self.expected_p = np.array([[[101, 101, 241],
                                     [138, 138, 164],
                                     [202, 202,   9]],
                                    [[ 93,  93,   8],
                                     [165, 165, 225],
                                     [191, 191, 156]],
                                    [[214, 214, 139],
                                     [158, 158, 134],
                                     [ 25,  25, 240]]], dtype=np.uint8)
        self.expected_d = np.array([[[ 73,  73, 246],
                                     [123, 123, 166],
                                     [158, 157,  16]],
                                    [[134, 133,   1],
                                     [183, 183, 222],
                                     [165, 165, 161]],
                                    [[218, 218, 138],
                                     [180, 180, 130],
                                     [ 96,  95, 228]]], dtype=np.uint8)
        self.expected_t = np.array([[[ 29, 107, 107],
                                     [ 97, 142, 141],
                                     [ 78, 211, 211]],
                                    [[201,  84,  84],
                                     [215, 162, 162],
                                     [119, 197, 197]],
                                    [[222, 213, 213],
                                     [216, 154, 154],
                                     [220,  11,  11]]], dtype=np.uint8)
        
    def test_rgb2gray_human_eye(self):
        hu_eye = diagfig.helpers.rgb2gray_human_eye(self.mat4test)
        assert_array_almost_equal(self.expected_hu_eye, hu_eye)
    def test_rgb2gray_digital(self):
        dig = diagfig.helpers.rgb2gray_digital(self.mat4test)
        assert_array_almost_equal(self.expected_dig, dig)
    def test_rgb_to_lms(self):
        rgb_to = diagfig.helpers.rgb_to_lms(self.mat4test)
        assert_array_almost_equal(self.expected_rgb_to, rgb_to)
    def test_lms_to_rgb(self):
        lms_to = diagfig.helpers.lms_to_rgb(self.mat4test)
        assert_array_almost_equal(self.expected_lms_to, lms_to)
    def test_rbg_to_rgb(self):
        back_to = diagfig.helpers.lms_to_rgb(diagfig.helpers.rgb_to_lms(self.mat4test))
        assert_array_almost_equal(self.mat4test, back_to)
    def test_simulate_p(self):
        sim_p = diagfig.helpers.simulate_colorblindness(self.mat4test * 256, "p")
        assert_array_almost_equal(self.expected_p, sim_p)
    def test_simulate_d(self):
        sim_d = diagfig.helpers.simulate_colorblindness(self.mat4test * 256, "d")
        assert_array_almost_equal(self.expected_d, sim_d)
    def test_simulate_t(self):
        sim_t = diagfig.helpers.simulate_colorblindness(self.mat4test * 256, "t")
        assert_array_almost_equal(self.expected_t, sim_t)
