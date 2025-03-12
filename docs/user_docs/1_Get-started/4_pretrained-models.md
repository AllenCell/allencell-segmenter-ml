# Pre-trained models

Users can jump start the process by applying one of the pre-trained models we provide to their data using the plugin, evaluate the segmentation results, and then further train these models to better suit their specific datasets.

Below is a table listing available pre-trained deep-learning models provided by the Allen Institute for Cell Science available for download from the plugin through the `Download` dialog

```{eval-rst}
.. list-table:: **Available pre-trained models by the Allen Institute for Cell Science**
    :widths: 10 10 50 50
    :header-rows: 1

    * - **Category**
      - Name
      - MegaSeg_v1
      - MegaSeg_light
    * - **Model Properties**
      - Description
      - CNN-based structure-agnostic 3D segmentation model, trained on 2600 open 3D fluorescent microscopy images of cellular structures in human hiPSCs
      - Lighter version of MegaSeg which compromises accuracy for speed
    * -
      - Architecture
      - nnUNet
      - small nnUNet with fewer kernels
    * -
      - Loss function
      - Generalized dice focal loss
      - Generalized dice focal loss
    * -
      - Optimization
      - Adam
      - Adam
    * -
      - # of Epochs
      - 1000
      - 1000
    * -
      - Stopping Criteria
      - Validation loss not improving for 100 epochs
      - Validation loss not improving for 100 epochs
    * -
      - System Trained On
      - Nvidia A100
      - Nvidia A100
    * -
      - Validation Step
      - -
      - -
    * - **Dependencies**
      - CytoDL Version
      - 1.7.1
      - 1.7.1
    * -
      - PyTorch Version
      - 2.4.0+cu118
      - 2.4.0+cu118
    * - **Training Data**
      - Image Resolution
      - 55x624x924; 60x624x924; 65x600x900; 65x624x924; 70x624x924; 75x624x924; 75x600x900
      - 55x624x924; 60x624x924; 65x600x900; 65x624x924; 70x624x924; 75x624x924; 75x600x900
    * -
      - Microscope Objective
      - 100x
      - 100x
    * -
      - Microscopy Technique
      - Spinning disk confocal
      - Spinning disk confocal
    * -
      - Public Data Link
      - `Dataset Link <https://open.quiltdata.com/b/allencell/tree/aics/hipsc_single_cell_image_dataset/>`_
      - `Dataset Link <https://open.quiltdata.com/b/allencell/tree/aics/hipsc_single_cell_image_dataset/>`_
    * -
      - Expected Performance
      - On NVIDIA-A100, 80GB, Inference @ 6.01 Secs for an Input image of size 924x624x65
      - On NVIDIA-A100, 80GB, Inference @ 2.32 Secs for an Input image of size 924x624x65
    * -
      - Structures Trained On
      - Actin bundles, ER(SERCA2), Adherens junctions, Desmosomes, Gap junctions, Myosin, Nuclear pores, Endosomes, ER (SEC61 Beta), Nuclear speckles, Golgi, Tight junctions, Mitochondria
      - Actin bundles, ER(SERCA2), Adherens junctions, Desmosomes, Gap junctions, Myosin, Nuclear pores, Endosomes, ER (SEC61 Beta), Nuclear speckles, Golgi, Tight junctions, Mitochondria
    * - **Inference Data**
      - Minimum Image Dimension
      - 16x16x16
      - 16x16x16



```
