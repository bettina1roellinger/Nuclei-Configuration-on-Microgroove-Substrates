# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 14:58:22 2023

@author: betti
"""
from functions_processing import *






folder_path = r'C:\path\to\directory\*.tif'





treament_folder(folder_path, channel=None, saving=True, saving_path=None, uses_left_clipping=False, uses_clipping=False, 
                      clip_limit=None, clip_value=None, quantile_frac=.985, uses_clahe=True, clahe_clip=0.01, kernel_size=None, nbins=256, 
                      uses_rm_bg=False, rm_bg_radius=15, uses_bilateral=False, rso_quantile=.99, rso_min_size=100, uses_sigmoid=False, 
                      cutoff=.2, gain=3, uses_gamma=False, gamma=2, uses_min_filter=False, min_size=2, uses_median_filter=False, 
                      median_size=7, uses_kb=False, radius=2, verbose=True)

