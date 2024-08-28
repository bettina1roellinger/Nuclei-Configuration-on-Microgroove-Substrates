#####################################################################################################################################
### Import
#####################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage, signal
import seaborn as sns
sns.set_theme()
from skimage import morphology, measure, segmentation, exposure, io, metrics, filters, transform
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
import logging
import math
#from cellpose import metrics as met
from glob import glob
import os
from matplotlib.patches import Rectangle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.cluster import KMeans
#import cellpose
#from cellpose import models
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
import plotly
import plotly.graph_objs as go
import plotly.express as px
# initialize plotly for jupyter notebook
plotly.offline.init_notebook_mode()

#####################################################################################################################################
### Utils for image loading, manipulation and display
#####################################################################################################################################

def load_images_from_path(path):
    elts = glob(path)
    elts_open = list(map(io.imread, elts))
    return elts, elts_open

def crop(im, row_min, row_max, col_min, col_max):
    return im[row_min:row_max, col_min:col_max]

#####################################################################################################################################
### Preprocessing
#####################################################################################################################################

### Preprocessing function

def treatment_dapi(
        img, print_img=False, uses_left_clipping=False, uses_clipping=False, clip_limit=None, clip_value=None,
        quantile_frac=.985, uses_clahe=True, clahe_clip=.01, kernel_size=None, nbins=256, uses_rm_bg=False,
        rm_bg_radius=15, uses_bilateral=False, rso_quantile=.99, rso_min_size=100, uses_sigmoid=True,  cutoff=.05,
        gain=1, uses_gamma=False, gamma=2, uses_min_filter=False, min_size=2, uses_median_filter=False, median_size=7,
        uses_kb=False, radius=2, verbose=True
        ):
    '''

    :param img: image with a single channel : the dapi channel
    :param print_img: boolean, if True, additionally prints the modified image
    default : False
    :param uses_left_clipping: boolean, if True, uses a clipping on the image small intensities (to reduce the
    background intensity variation)
    default : False
    :param uses_clipping: boolean, if True, uses a clipping on the image intensities (to remove high abnormal
    intensities, i.e. intensity outliers)
    default : False (previously True)
    :param clip_limit: integer or float, maximal intensity in the clipped image (i.e. the value at which we are
    clipping)
    default : None
    :param clip_value: integer or float, value to which the clipped pixels are set
    default : None
    :param quantile_frac: float between 0 and 1, quantile to compute
    default : .985
    :param uses_clahe: boolean, if True, uses a CLAHE adaptative contrast enhancement
    default : True
    :param clahe_clip: float, clip limit of the CLAHE filter
    default : .01  #.03 after Bettina reevaluation
    :param kernel_size: CLAHE kernel_size, int or array_like
    If None, is set to img.shape[0]//64  # 128
    default : None
    :param nbins: CLAHE nbins, int
    default : 256
    :param uses_rm_bg: boolean, if True, removes the background using morphological operations
    If False, a denoise_tv_chambolle filter is used.
    default : False
    :param rm_bg_radius: int, radius of the morphological element used in the closing operation
    default : 15
    :param uses_bilateral: boolean, if True (and uses_rm_bg==False), a bilateral filter is applied.
    default : False
    :param rso_quantile: float between 0 and 1, the quantile used for the threshold to isolate the deformed nuclei centers
    default : .99
    :param rso_min_size: int, the minimum size of the connexe component kept after the remove_small_objects operation,
    while isolating the deformed nuclei centers
    default : 100
    :param uses_sigmoid: boolean, if True, uses a Sigmoid transformation of the intensities (must be False for this
    function to return a gamma transformed image, if uses_gamma=True)
    default : True
    :param cutoff: float, cutoff value of the Sigmoid filter
    default : .3 (previously .7)
    :param gain: integer or float, value of the gain of the Sigmoid filter
    default : 6 (previously 7)
    :param uses_gamma: boolean, if True, uses a Gamma transformation of the intensities (must be True and
    uses_sigmoid=False for this function to return a gamma transformed image)
    default : False
    :param gamma: integer of float, value of the Gamma filter parameter
    default : 2
    :param uses_min_filter: boolean, if True, uses a Min filter of size min_size
    default : True
    :param min_size: integer or float, size of the Min filter
    default : 2
    :param uses_median_filter: boolean, if True, uses a Median filter of size median_size
    default : True
    :param median_size: integer or float, size of the Median filter
    default : 7
    :param uses_kb: boolean, if True, uses a KB filter of with radius radius
    default : True
    :param radius: integer, size of the KB filter
    default : 2 (previously 5)
    :param verbose: boolean, if True, prints some info on the successive treatments
    default : True
    :return: img, a single one channel image
    '''

    # Exceptions
    if not isinstance(img, np.ndarray) and len(img.shape) > 2:
        logging.exception(
            "treatment_dapi parameter 'img' must be a one dimension numpy image (ndarray)")
        raise TypeError(
            "'img' must be a numpy image (ndarray)"
        )

    if not isinstance(uses_left_clipping, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_left_clipping' must be a boolean")
        raise TypeError(
            "'uses_left_clipping' must be a boolean"
        )

    if not isinstance(uses_clipping, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_clipping' must be a boolean")
        raise TypeError(
            "'uses_clipping' must be a boolean"
        )

    if not isinstance(uses_clahe, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_clahe' must be a boolean")
        raise TypeError(
            "'uses_clahe' must be a boolean"
        )

    if not isinstance(uses_rm_bg, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_rm_bg' must be a boolean")
        raise TypeError(
            "'uses_rm_bg' must be a boolean"
        )

    if not isinstance(uses_bilateral, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_bilateral' must be a boolean")
        raise TypeError(
            "'uses_bilateral' must be a boolean"
        )

    if not isinstance(uses_sigmoid, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_sigmoid' must be a boolean")
        raise TypeError(
            "'uses_sigmoid' must be a boolean"
        )

    if not isinstance(uses_gamma, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_gamma' must be a boolean")
        raise TypeError(
            "'uses_gamma' must be a boolean"
        )

    if not isinstance(uses_min_filter, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_min_filter' must be a boolean")
        raise TypeError(
            "'uses_min_filter' must be a boolean"
        )

    if not isinstance(uses_median_filter, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_median_filter' must be a boolean")
        raise TypeError(
            "'uses_median_filter' must be a boolean"
        )

    if not isinstance(uses_kb, bool):
        logging.exception(
            "treatment_dapi parameter 'uses_kb' must be a boolean")
        raise TypeError(
            "'uses_kb' must be a boolean"
        )

    if not isinstance(clip_limit, int) and not isinstance(clip_limit, float) and clip_limit is not None:
        logging.exception(
            "treatment_dapi parameter 'clip_limit' must be an integer or a float")
        raise TypeError(
            "'clip_limit' must be an integer or a float"
        )

    if clip_value is not None and not isinstance(clip_value, int) and not isinstance(clip_value, float):
        logging.exception(
            "treatment_dapi parameter 'clip_value' must be an integer or a float")
        raise TypeError(
            "'clip_value' must be an integer or a float"
        )

    if kernel_size is not None and not isinstance(kernel_size, int) and not isinstance(kernel_size, float):
        logging.exception(
            "treatment_dapi parameter 'kernel_size' must be an integer or a float")
        raise TypeError(
            "'kernel_size' must be an integer or a float"
        )

    if not isinstance(quantile_frac, float) and not(quantile_frac >= 0) and not(quantile_frac <= 1):
        logging.exception(
            "treatment_dapi parameter 'quantile_frac' must be a float between 0 and 1")
        raise TypeError(
            "'quantile_frac' must be a float between 0 and 1"
        )

    if not isinstance(radius, int):
        logging.exception(
            "treatment_dapi parameter 'radius' must be an integer")
        raise TypeError(
            "'radius' must be an integer"
        )

    if not isinstance(nbins, int):
        logging.exception(
            "treatment_dapi parameter 'nbins' must be an integer")
        raise TypeError(
            "'nbins' must be an integer"
        )

    if not isinstance(clahe_clip, float):
        logging.exception(
            "treatment_dapi parameter 'clahe_clip' must be a float")
        raise TypeError(
            "'clahe_clip' must be a float"
        )

    if not isinstance(cutoff, float):
        logging.exception(
            "treatment_dapi parameter 'cutoff' must be a float")
        raise TypeError(
            "'cutoff' must be a float"
        )

    if not isinstance(gain, float) and not isinstance(gain, int):
        logging.exception(
            "treatment_dapi parameter 'gain' must be an integer or a float")
        raise TypeError(
            "'gain' must be an integer or a float"
        )

    if not isinstance(gamma, float) and not isinstance(gamma, int):
        logging.exception(
            "treatment_dapi parameter 'gamma' must be an integer or a float")
        raise TypeError(
            "'gamma' must be an integer or a float"
        )

    if not isinstance(min_size, float) and not isinstance(min_size, int):
        logging.exception(
            "treatment_dapi parameter 'min_size' must be an integer or a float")
        raise TypeError(
            "'min_size' must be an integer or a float"
        )

    if not isinstance(median_size, float) and not isinstance(median_size, int):
        logging.exception(
            "treatment_dapi parameter 'median_size' must be an integer or a float")
        raise TypeError(
            "'median_size' must be an integer or a float"
        )

    def __kb(I, r):
        """
        Elementary Kramer/Bruckner filter. Also called toggle filter.
        I: image
        r: radius of structuring element (disk), for max/min evaluation
        """
        se = morphology.disk(r)
        D = morphology.dilation(I, footprint=se)
        E = morphology.erosion(I, footprint=se)
        difbool = D - I < I - E
        k = D * difbool + E * (~difbool)
        return k

    def __show_mod(im, ini):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7))
        pl1 = axes[0].imshow(im, cmap='gray')
        pl2 = axes[1].imshow(ini, cmap='gray')
        axes[0].title.set_text('modified image')
        axes[1].title.set_text('initial image')
        axes[0].grid(False)
        axes[1].grid(False)
        fig.colorbar(pl1, ax=axes[0])
        fig.colorbar(pl2, ax=axes[1])
        fig.tight_layout()

    def __rm_background(im, rm_bg_radius=15, rso_quantile=.99, rso_min_size=100):
        #rect = morphology.rectangle(nrows=5, ncols=20)
        disk = morphology.disk(rm_bg_radius)
        im_mod = morphology.closing(im, footprint=disk)
        im_op = morphology.black_tophat(im, footprint=disk)
        im_op_th = morphology.remove_small_objects(
            im_op > np.quantile(im_op.flatten(), rso_quantile),
            min_size=rso_min_size
        )
        im_mod_c = im_mod.copy()
        im_mod_c[im_op_th] = im[im_op_th]
        return im_mod_c

    # Function
    if verbose:
        print("Dapi image treatment :\n")
    if uses_clipping:
        if clip_limit is None:
            clip_limit = np.quantile(img.flatten(), quantile_frac)
        if clip_value is None:
            clip_value = clip_limit
        if verbose:
            print("Clipping of intensities > {} to {}\n".format(clip_limit, clip_value))
    if uses_clahe:
        if kernel_size is None:
            kernel_size = img.shape[0]//64  # 128
        if verbose:
            print("CLAHE with clip_limit = {}, kernel_size = {}, nbins = {}\n".format(clahe_clip, kernel_size, nbins))
    if uses_rm_bg:
        print(f"Removing background with a closing operation with disk({rm_bg_radius}), "
              f"a blacktophat with the same element, a threshold at quantile {rso_quantile} "
              f"and with min connected component size {rso_min_size}\n")
    elif not uses_rm_bg and not uses_bilateral:
        print(f"Denoising using a denoise_tv_chambolle(weight=0.05) filter \n")
    else:
        print(f"Denoising using a denoise_bilateral(win_size=None, sigma_spatial=8) filter \n")
    if uses_sigmoid and verbose:
        print("Sigmoid with cutoff={} and gain={}\n".format(cutoff, gain))
        print("To get a gamma transformed image, set uses_gamma=True and uses_sigmoid=False\n")
    if uses_gamma and not uses_sigmoid and verbose:
        print("Gamma with gamma={}\n".format(gamma))
    if uses_min_filter and verbose:
        print("Min filter of size {}\n".format(min_size))
    if uses_median_filter and verbose:
        print("Median filter of size {}\n".format(median_size))
    if uses_kb and verbose:
        print("KB filter with radius {}\n".format(radius))
    # Clipping to 3000
    img_copy = img.copy()

    if uses_left_clipping:
        img_copy[img_copy < np.quantile(img.flatten(), .2)] = 0

    if uses_clipping:
        img_copy[img_copy > clip_limit] = clip_value

    # CLAHE
    if uses_clahe:
        img_copy_eq = exposure.equalize_adapthist(img_copy, kernel_size=kernel_size, clip_limit=clahe_clip, nbins=nbins)
    else:
        img_copy_eq = img_copy

    # Remove background
    if uses_rm_bg:
        img_copy_eq = __rm_background(
            img_copy_eq,
            rm_bg_radius=rm_bg_radius,
            rso_quantile=rso_quantile,
            rso_min_size=rso_min_size
        )
    elif not uses_rm_bg and not uses_bilateral:
        img_copy_eq = denoise_tv_chambolle(img_copy_eq, weight=0.05)
    else:
        img_copy_eq = denoise_bilateral(img_copy_eq, win_size=None, sigma_spatial=8)

    # Sigmoid and Gamma transform
    if uses_sigmoid:
        I_mod = exposure.adjust_sigmoid(img_copy_eq, cutoff=cutoff, gain=gain)
    elif uses_gamma:
        I_mod = exposure.adjust_gamma(img_copy_eq, gamma=gamma)
    else:
        I_mod = img_copy_eq

    # Min, Median and KB filters
    if not uses_min_filter and not uses_median_filter and not uses_kb:
        if print_img:
            __show_mod(I_mod, img)
        return I_mod
    else:
        if uses_min_filter:
            I_mod = ndimage.minimum_filter(I_mod, min_size)
        if uses_median_filter:
            I_mod = ndimage.median_filter(I_mod, size=median_size)
        if uses_kb:
            I_mod = __kb(I_mod, radius)
        if print_img:
            __show_mod(I_mod, img)
        return I_mod
    

### Preprocessing function on folder
    
def treament_folder(folder_path, channel=None, saving=False, saving_path=None, uses_left_clipping=False, uses_clipping=False, 
                    clip_limit=None, clip_value=None, quantile_frac=.985, uses_clahe=True, clahe_clip=.03, kernel_size=None, nbins=256, 
                    uses_rm_bg=False, rm_bg_radius=15, uses_bilateral=False, rso_quantile=.99, rso_min_size=100, uses_sigmoid=True, 
                    cutoff=.3, gain=6, uses_gamma=False, gamma=2, uses_min_filter=False, min_size=2, uses_median_filter=False, 
                    median_size=7, uses_kb=False, radius=2, verbose=True):
    """
    folder_path: string, path to the specific folder of images to process: "path/to/images/*"
    channel: int, channel to use (default: None, takes the image as a 1-channeled image)
    saving: bool, saving the images into a folder, or returning them
    saving_path: string, path to saving: "path/to/saving/"
    """
    
    elts = glob(folder_path)
    elts_open = list(map(io.imread, elts))
    if channel is not None:
        elts_open = [im[:,:,channel] for im in elts_open]
        
    elts_treated = []
    for img in elts_open:
        elts_treated.append(
            treatment_dapi(
                img, 
                uses_left_clipping=uses_left_clipping, 
                uses_clipping=uses_clipping, 
                clip_limit=clip_limit, 
                clip_value=clip_value,
                quantile_frac=quantile_frac, 
                uses_clahe=uses_clahe, 
                clahe_clip=clahe_clip, 
                kernel_size=kernel_size, 
                nbins=nbins, 
                uses_rm_bg=uses_rm_bg,
                rm_bg_radius=rm_bg_radius, 
                uses_bilateral=uses_bilateral, 
                rso_quantile=rso_quantile, 
                rso_min_size=rso_min_size, 
                uses_sigmoid=uses_sigmoid,  
                cutoff=cutoff,
                gain=gain, 
                uses_gamma=uses_gamma, 
                gamma=gamma, 
                uses_min_filter=uses_min_filter, 
                min_size=min_size, 
                uses_median_filter=uses_median_filter, 
                median_size=median_size,
                uses_kb=uses_kb, 
                radius=radius, 
                verbose=verbose
            )
        )
    
    if saving:
        if saving_path is not None:
            for img_index in range(len(elts_treated)):
                img = elts_treated[img_index]
                io.imsave(saving_path+elts[img_index][len(folder_path)-1:], img)             
        else:
            for img_index in range(len(elts_treated)):
                img = elts_treated[img_index]
                saving_path = folder_path[:-5]+"treated_images/"
                io.imsave(saving_path+elts[img_index][len(folder_path)-5:], img)  
    else:
        return elts_treated
    
    
### Metrics

def snr(im, th):
    m = np.mean(im.flatten())
    sd = np.std(im[im < th])
    return m/sd

def psnr(img, img_mod):
    return metrics.peak_signal_noise_ratio(img_mod, img)

def ssim(im, im_preprocessed):
    return metrics.structural_similarity(im, im_preprocessed)

def norm_grad(im):
    """
    Note: This function implements my home-made preprocessing evaluation metric, which is not really efficient. Use other metrics instead. 
    """
    return np.linalg.norm(signal.convolve2d(im, [[-1,1]], mode='same') + signal.convolve2d(im, [[-1],[1]], mode='same'))

def cnr(I, h=None, disp_max=False, verbose=False, threshold=False):
    """
    Note: This metric is the more useful one with the analysed images. However, it should be compined with SNR. 
    You can also try cnr_bettina metric, which is the same metric under another implementation
    """
    def __mean_nuclei(I, h, disp_max, verbose):
        peaks = morphology.h_maxima(I, h=h, footprint=None)
        if verbose:
            print(f"nb local max detected: {np.sum(peaks)}")
        if disp_max:
            # to display the image with its local maxima
            plt.imshow(I, cmap='gray')
            x, y = np.where(peaks.T == True)
            plt.scatter(x, y, marker='*', alpha=.2)
            plt.grid(False)
            plt.show()
        if np.sum(peaks) == 0:  # no nuclei detected in the image
            m = 0
        else:
            m = np.mean(I[peaks == True]) # mean of the intensities of the local maxima
        return m
    
    def __mean_bg(I):
        hist, bin_centers = exposure.histogram(I, nbins=10000)
        mbg = bin_centers[np.argmax(hist)]
        return mbg

    def __somme_var(I, threshold):
        if threshold:
            th = np.quantile(I.flatten(), .95)
            return np.sqrt(np.var(I[I < th]) + np.var(I[I > th]))
        else:
            disk = morphology.disk(15)
            I_bth = morphology.black_tophat(I, footprint=disk) #opening closing white_tophat black_tophat
            #plt.imshow(I_bth, cmap='gray')
            #plt.grid(False)
            #plt.show()
            #print(np.quantile(I_bth.flatten(), .99))
            #print(np.var(I_bth))
            #print(np.var(I[I_bth > np.quantile(I_bth.flatten(), .995)]))
            #print(np.var(I[~np.logical_and(I, I_bth)]))
            return np.sqrt(
                np.var(I_bth) + np.var(I[~np.logical_and(I, I_bth)])
            )
    
    if h is None:
        h = np.mean(I.flatten())

    m = __mean_nuclei(I, h, disp_max, verbose)
    mbg = __mean_bg(I)
    stdbg = __somme_var(I, threshold=threshold)
    if verbose:
        print("m: ", m, "mbg: ", mbg, "std: ", stdbg)
    cnr_val = np.abs(m - mbg) / stdbg
    return cnr_val



def cnr_bettina(I, h=None, disp_max=False, verbose=False, threshold=False):
    
    def __mean_nuclei_bettina(I, h, disp_max, verbose):
        peaks = morphology.h_maxima(I, h=h, footprint=None)
        if verbose:
            print(f"nb local max detected: {np.sum(peaks)}")
        if disp_max:
            # to display the image with its local maxima
            plt.imshow(I, cmap='gray')
            x, y = np.where(peaks.T == True)
            plt.scatter(x, y, marker='*', alpha=.2)
            plt.grid(False)
            plt.show()
        if np.sum(peaks) == 0:  # no nuclei detected in the image
            m = 0
            n = 0
        else:
            m = np.mean(I[peaks == True]) # mean of the intensities of the local maxima
            n = np.std(I[peaks == True]) # mean of the intensities of the local maxima
        return m,n
    
    def __mean_bg(I):
        hist, bin_centers = exposure.histogram(I, nbins=10000)
        mbg = bin_centers[np.argmax(hist)]
        return mbg

    def __somme_var(I, threshold):
        if threshold:
            th = np.quantile(I.flatten(), .95)
            return np.sqrt(np.var(I[I < th]) + np.var(I[I > th]))
        else:
            disk = morphology.disk(15)
            I_bth = morphology.black_tophat(I, footprint=disk) #opening closing white_tophat black_tophat
            #plt.imshow(I_bth, cmap='gray')
            #plt.grid(False)
            #plt.show()
            #print(np.quantile(I_bth.flatten(), .99))
            #print(np.var(I_bth))
            #print(np.var(I[I_bth > np.quantile(I_bth.flatten(), .995)]))
            #print(np.var(I[~np.logical_and(I, I_bth)]))
            return np.sqrt(
                np.var(I_bth) + np.var(I[~np.logical_and(I, I_bth)])
            )
    
    if h is None:
        h = np.mean(I.flatten())

    m, n = __mean_nuclei_bettina(I, h, disp_max, verbose)
    mbg = __mean_bg(I)
    stdbg = __somme_var(I, threshold=threshold)
    if verbose:
        print("m: ", m, "mbg: ", mbg, "std: ", stdbg, "n: ", n)
    cnr_val = (np.abs(m - mbg)) / (np.sqrt(stdbg**2+n**2))
    return cnr_val


def colormap_func(im, func=cnr, kernel_size=30):
    """
    Display a heatmap of the metric func applied on a moving kernel of size kernel_size on the image im
    """
    nx, ny = im.shape
    color_im = np.zeros(im.shape)
    for i in range(nx//kernel_size):
        for j in range(ny//kernel_size):
            color_im[
                i*kernel_size : (i+1)*kernel_size,
                j*kernel_size:(j+1)*kernel_size
            ] += func(
                im[
                    i*kernel_size : (i+1)*kernel_size,
                    j*kernel_size : (j+1)*kernel_size]
            )

    pl = plt.imshow(color_im)
    plt.colorbar(pl)
    plt.grid(False)
    plt.show()

#####################################################################################################################################
### Segmentation
#####################################################################################################################################

def compute_masks(dataset, dataset_names, model_path, saving_path):
    # load custom model
    model = models.CellposeModel(pretrained_model=model_path)

    # segment cells using the custom model
    masks, flows, styles = model.eval(dataset, diameter=None, channels=[0, 0])

    # save masks as PNG files
    cellpose.io.save_masks(dataset, masks, flows, dataset_names, png=True, savedir=saving_path)
    
    
def load_cellpose_model(model_path):
    model = models.CellposeModel(pretrained_model=model_path)
    return model

def compute_lamin_masks_li(dataset, dataset_names, saving_path):
    masks = [
        measure.label(
            segmentation.clear_border(
                image > filters.threshold_li(image)
            )
        ) 
        for image in dataset
    ]
    # save masks as PNG files
    for i in range(len(masks)):
        io.imsave(saving_path[:-1]+dataset_names[i].replace('/', ';')+'.png', masks[i])

def compute_lamin_masks_li_br(dataset, saving_path):
    masks = [
        measure.label(
            segmentation.clear_border(
                image > filters.threshold_li(image)
            )
        ) 
        for image in dataset
    ]
    
    # save masks as PNG files
    for i in range(len(masks)):
        io.imsave(saving_path + str(i) + '.png', masks[i])

def compute_lamin_masks_otsu_br(dataset, saving_path):
    masks = [
        measure.label(
            segmentation.clear_border(
                image > filters.threshold_otsu(image)
            )
        ) 
        for image in dataset
    ]
    
    # save masks as PNG files
    for i in range(len(masks)):
        io.imsave(saving_path + str(i) + '.png', masks[i])

#####################################################################################################################################
### CROPS
#####################################################################################################################################

       
def plot_bbox(image, labeled_mask):
    props = measure.regionprops(labeled_mask)
    bboxs = []
    for prop in props:
        bboxs.append([prop.label, prop.bbox])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    ax.grid(False)
    for bbox in bboxs:
        label = bbox[0]
        min_row, min_col, max_row, max_col = bbox[1]
        ax.add_patch(
            Rectangle(
                (min_col, min_row),
                width=max_col-min_col,
                height=max_row-min_row,
                fc ='none',
                ec ='r',
                lw = 1
            )
        )
        ax.annotate(
            f'{label}',
            (min_col, min_row),
            fontsize=6,
            fontweight="bold",
            color="white"
        )
    fig.show()


def resize_and_save(img, mask, path_to_saving, new_shape, area_th=700): 
    
    def __crop_resize(label_crop, new_shape):
        new_s = np.min(new_shape)
        shapes = np.array(label_crop.shape)
        max_dim = np.max(shapes)
        label_crop = transform.resize(label_crop, output_shape=tuple([int(elt) for elt in new_s * (shapes/max_dim)]), mode='edge')
        # label_crop = transform.rescale(label_crop, scale=new_s/max_dim, mode='edge')
        shapes = np.array(label_crop.shape)
        delta_y = (new_shape[1] - shapes[1]) // 2
        delta_x = (new_shape[0] - shapes[0]) // 2
        label_crop = np.pad(label_crop,
                            ((delta_x, new_shape[0]-(delta_x + shapes[0])), (delta_y, new_shape[1]-(delta_y + shapes[1]))),
                            constant_values = np.min(label_crop.flatten())
                           )
        return label_crop
    
    print(f"Saving crops with area > {area_th}")
    props = measure.regionprops(mask)
    bboxs = []
    for prop in props:
        label = prop.label
        min_row, min_col, max_row, max_col = prop.bbox
        if prop.area > area_th: # saving only the not too small nuclei
            label_crop = __crop_resize(crop(img, min_row, max_row, min_col, max_col), new_shape)
            io.imsave(path_to_saving + str(label) + '.png', label_crop)
            io.imsave(path_to_saving + str(label) + '_flipy.png', np.fliplr(label_crop))
            io.imsave(path_to_saving + str(label) + '_flipx.png', np.flipud(label_crop))
            io.imsave(path_to_saving + str(label) + '_rot.png', np.flipud(np.fliplr(label_crop)))
        #io.imsave(path_to_saving + str(label) + '_rot.png', transform.rotate(label_crop, angle=90))
    print(f"All labels saved to {path_to_saving}")
    
def create_and_save_crops(lamin_preprocessed_path, masks_path, crop_size=64, saving_path=""):
    masks_names, masks = load_images_from_path(masks_path)
    lamin_preprocessed_names, lamin_preprocessed = load_images_from_path(lamin_preprocessed_path)

    for i in range(len(lamin_preprocessed)):
        
        path = lamin_preprocessed_names[i][len(lamin_preprocessed_path)-1:]+';'
        
        if saving_path != "":
            path = saving_path + path
                
        resize_and_save(
            lamin_preprocessed[i], 
            masks[i],
            path, 
            (crop_size,crop_size)
        ) 
    
    

    
    
    
    
    

    
    