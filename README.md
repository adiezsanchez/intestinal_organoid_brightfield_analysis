<h1>Intestinal organoid brightfield analysis</h1>

This repository contains a number of tools to speed up the interpretation and analysis of images from intestinal organoids acquired using a brightfield microscope. In our case an EVOS M7000 multiwell scanner which outputs the following filenames: P1_Plate_M_p00_z00_0_A01f00d0. The scripts use the previously mentioned naming convention to extract the well_id from each image ("A01"), scan through all z-planes ("z00") and find the focal plane with the most organoids in focus. Then it generates a plate view of the entire multiwell plate at high resolution for data exploration.

This is a work in progress so I will be gradually including functionalities (feature extraction (morphology, nr of organoids per well) and object classification using deep learning)

<h1>Instructions</h1>

1. In order to run these Jupyter notebooks you will need to familiarize yourself with the use of Python virtual environments using Mamba. See instructions [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html).

2. Then you will need to create a virtual environment using the following command:

<code>mamba create -n devbio-napari devbio-napari python=3.9 pyqt -c conda-forge</code>
