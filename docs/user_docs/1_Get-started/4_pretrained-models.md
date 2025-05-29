# Pre-trained models

Users can jump start the process of creating a deep-learning model by:
- applying one of the provided pre-trained models to their data using the plugin
- evaluating the segmentation results
- fine-tuning these models to better suit their specific datasets

Below is a table listing available pre-trained models available for download from the plugin through the {bdg-dark-line}`Download` dialog under the {bdg-dark}`Help` button. If you use our pre-trained models in your own research, please cite us; citation info included in the table. Thank you!

```{eval-rst}
.. list-table:: **Available pre-trained models by the Allen Institute for Cell Science**
    :widths: 10 30 80
    :header-rows: 1

    * - **Category**
      - Name
      - MegaSeg_v1
    * - **Model Properties**
      - Description
      - CNN-based structure-agnostic 3D segmentation model, trained on 2600 open 3D fluorescent microscopy images of cellular structures in human hiPSCs
    * -
      - Architecture
      - nnUNet
    * -
      - Loss function
      - Generalized dice focal loss
    * -
      - Optimization
      - Adam
    * -
      - # of Epochs
      - 1000
    * -
      - Stopping Criteria
      - Validation loss not improving for 100 epochs
    * -
      - System Trained On
      - Nvidia A100
    * - **Dependencies**
      - CytoDL Version
      - 1.7.1
    * -
      - PyTorch Version
      - 2.4.0+cu118
    * - **Training Data**
      - Image Resolution
      - 55x624x924; 60x624x924; 65x600x900; 65x624x924; 70x624x924; 75x624x924; 75x600x900
    * -
      - Microscope Objective
      - 100x
    * -
      - Microscopy Technique
      - Spinning disk confocal
    * -
      - Public Data Link
      - `Dataset Link <https://open.quiltdata.com/b/allencell/tree/aics/hipsc_single_cell_image_dataset/>`_
    * -
      - Expected Performance
      - On NVIDIA-A100, 80GB, Inference @ 6.01 Secs for an Input image of size 924x624x65
    * -
      - Structures Trained On
      - Actin bundles, ER(SERCA2), Adherens junctions, Desmosomes, Gap junctions, Myosin, Nuclear pores, Endosomes, ER (SEC61 Beta), Nuclear speckles, Golgi, Tight junctions, Mitochondria
    * - **Inference Data**
      - Minimum Image Dimension
      - 16x16x16
    * - 
      - Threshold value
      - We have validated MegaSeg using a 50% threshold or the threshold value of 128 i.e., pixels for which the model output is lower than 128 will be classified as background.
    * - **Disclosure**
      - Info
      - Model will not be able to segment the tops and bottoms of nuclear(Lamin) and plasma(CAAX) membrane. Model will also face issue in segmenting dataset specific attributes (e.g., filled vesciles on lysosomes) in contrast to general attributes.
    * - **Citation**
      - Info
      - The MegaSeg preprint is in preparation, if you use MegaSeg model in your own research, in the meantime you can cite `Segmenter ML Plugin <https://www.napari-hub.org/plugins/allencell-segmenter-ml>`_. Thank you!



```
