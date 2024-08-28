# Nuclei-Configuration-on-Microgroove-Substrates
This repository contains all the code (Python scripts and Jupyter Notebooks) needed to cluster greyscale images of nuclei on micropatterned substrates
# Project Overview
Various diseases including laminopathies and certain types of cancer are associated with abnormal nuclear mechanical properties that influence cellular and nuclear deformations in complex environments. Recently, microgrooves substrates designed to mimic the anisotropic topography of basement membrane have been shown to induce significant 3D nuclear deformations in various adherent cell types. Importantly, these deformations appear to be different in myoblast cells derived from laminopathy patients from those derived from normal individuals. Deep learning techniques encompassing image analysis have been widely used for medical application. Here we assess the ability of a variational autoencoder with gaussian mixture model to cluster the nuclei of myoblast cells with laminopathy-associated mutation on micropattern substrates. 
# Getting Started 
Before you begin make sure you have all the prerequisites installed on your system:

**1** Python: You'll need Python 3.7 for running the project.

**2** Pip: Make sure you have pip, the Python package manager, installed.
# Installation 
**1.Clone the repository** Start by cloning this GitHub repository to your local machine
**2.Create and activate a virtual environment** It's recommended to create a virtual environment to isolate your project dependencies
```python 
conda create -n myenv 
conda activate myenv
```
**3.Install dependencies** 

**4.Explore the project and the code** This project is organized into distinct parts that can be used independently. We first addressed image processing to prepare for subsequent deep learning approaches. You will find various scripts for processing the images. The main code consists of multiple functions that you can utilize individually, depending on your needs and the quality of your grayscale images. You will also find a Jupyter Notebook in this repository that implements a Variational Autoencoder and a Gaussian Mixture Model algorithm, applied to the MNIST dataset and a grayscale nuclei image dataset. 

