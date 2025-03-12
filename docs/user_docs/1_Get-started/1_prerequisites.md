# Prerequisites

(system-requirements)=

## A. System requirements

We currently support `Windows`, `MacOS`, and `Linux` operating systems. The minimum system requirements are:

- 8GB of RAM
- 8 CPU Cores
- System storage: at least the same amount of free space as the size of your training dataset
- 1 NVIDIA GPU with 8GB of VRAM (optional)

:::{tip}
A GPU is **highly recommended** for training models and make segmentation predictions.
:::

:::{note}
- If you plan to use the plugin without a GPU, the training will default to using your CPU and will be significantly slower
- Additionally, factors such as image size, 2D vs. 3D, resolution, model size, etc. may cause prediction to run slowly without a GPU
:::

______________________________________________________________________

(data-requirements)=

## B. Data requirements

To use the plugin effectively, you should have the training data as following:
- **Existing** high-quality segmentations, along with their corresponding raw original microscopy images
- Large number of training images are recommended for robust, accurate segmentation models
  - This number depends on cellular structure type and their counts wihtin a single FoV or file

The **Segmenter ML plugin** supports:
: - Image data format: `.tiff`, `.ome-tiff`
  - Dimensions: 2D or 3D, single- (recommended for faster processing) or multi-channel
  - Segmentation object: cellular structures (recommended)

:::{tip}
We recommend using the [Allen Cell & Structure Segmenter Classic plugin](https://www.napari-hub.org/plugins/napari-allencell-segmenter) to generate segmentations, as it ensures high-quality segmentations and compatibility with the required directory structure for this plugin.

If you would like to check out example data that works with the plugin, we share the data we have used for the pre-trained models provided with this plugin. Please refer to the {ref}`Pre-trained models` page for more details.
:::

______________________________________________________________________
