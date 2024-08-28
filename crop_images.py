# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:07:37 2023

@author: betti
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

from functions_processing import *
from tifffile import imread






####Here is the path leading to your images ( composite or geryscale)####
path = r"C:\path\to\directory\*.tif"

####To use the following function, you need to create a 'greyscale' folder in the same directory as your composite images####
#### Here is the path to your images (composite or greyscale) ####
#greyscale_image(path, channel=None, saving_path=None, saving=True)

#### Here is the path to the folder where you should place the binarized mask, named after the images ####
saving_path = r"C:\path\to\directory\mask"

images_names, images = load_images_from_path(path)

compute_lamin_masks_li_br(images, saving_path)

#### Make sure to delete the files in the folder before you run the program again ####
saving_path2 = r"C:\path\to\directory\*.png"
mask_names, labeled_mask = load_images_from_path(saving_path2)


for i in range (len(labeled_mask)):

    props = measure.regionprops(labeled_mask[i])
    bboxs = []
    label_ar=[]
    min_row_ar=[]
    max_row_ar=[]
    min_col_ar=[]
    max_col_ar=[]
    new_shape=(64,64)

    for prop in props:
        label = prop.label
        min_row, min_col, max_row, max_col = prop.bbox
        #label_ar.append(label)
        # min_row_ar, min_col, max_row, max_col
        # lab1=prop.bbox[0]
        lab_crop=crop(images[i], min_row, max_row, min_col, max_col)

# # fig, ax = plt.subplots()
# # plt.imshow(lab_crop, cmap='gray')
# # plt.grid(False)

    shapes = np.array(lab_crop.shape)
    print('shapes',shapes[0])
    max_dim = np.max(shapes)
    new_s = np.min(new_shape)
    for elt in new_s * (shapes/max_dim):
        print('elt',elt)


    path_to_saving= r"C:\path\to\directory\crops"+str(i)+'image' 
    resize_and_save(images[i], labeled_mask[i], path_to_saving,(64,64), area_th=700)





    
              
