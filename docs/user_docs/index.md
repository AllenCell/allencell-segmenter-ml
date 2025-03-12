% Segmenter ML plugin for napari documentation master file, created by
% sphinx-quickstart on Wed Nov  6 01:08:21 2024.

# Introduction


A [napari](http://napari.org) plugin for deep-learning based segmentation of cellular structures.

<br>

:::{figure} images/SegmenterML-plugin_fig1_output.png
:align: center
:width: 80%
:alt: schematic of the input & output of Segmenter ML plugin
:::

- **Available at no cost** --- available on PyPI
- **User-friendly** --- leverage *napari* as a fast 3D viewer with interactive plugin interface
- **Beginner-friendly** --- new to machine learning? This plugin simplifies the application of machine learning in the segmentation process through the 3 main modules:
    - **Curation**: curate training datasets
    - **Training**: iteratively train custom segmentation model(s) (UNET) to target cellular structure with wide morphological variability
    - **Prediction & Thresholding**: generate segmentation prediction on 2D and 3D cell image data

<br>

:::{figure} images/napari_anatomy.png
:alt: screenshot of napari with the Segmenter ML plugin
**Segmenter ML plugin** in *napari* viewer
::: 

***

## About this User Guide

This document provides an overview of key concepts behind the plugin, offers step-by-step instructions for complete workflows, and share valuable resources, such as available pre-trained models (MegaSeg) and open-access training datasets.



:::{toctree}
:hidden:
self
0_overview
:::

:::{toctree}
:hidden:
:caption: Main
1_Get-started/0_index_get-started
2_Workflows/0_index_workflows
3_How-do-i/0_index_how-do-i
:::

:::{toctree}
:hidden:
:caption: Support
4_Help/0_index_help
:::

% Indices and tables
% ==================
%
% * :ref:`genindex`
% * :ref:`modindex`
% * :ref:`search`
