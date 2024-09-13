# Allencell-ml-segmenter
<!-- 
TODO disabling while I figure out how to fix these values
[![License BSD-3](https://img.shields.io/pypi/l/allencell-ml-segmenter.svg?color=green)](https://github.com/AllenCell/allencell-ml-segmenter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/allencell-ml-segmenter.svg?color=green)](https://pypi.org/project/allencell-ml-segmenter)
[![Python Version](https://img.shields.io/pypi/pyversions/allencell-ml-segmenter.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/AllenCell/allencell-ml-segmenter/branch/main/graph/badge.svg?token=E976SiYFP6)](https://codecov.io/gh/AllenCell/allencell-ml-segmenter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/allencell-ml-segmenter)](https://napari-hub.org/plugins/allencell-ml-segmenter) 
-->
[![Tests](https://github.com/AllenCell/allencell-ml-segmenter/actions/workflows/test_lint.yaml/badge.svg)](https://github.com/AllenCell/allencell-ml-segmenter/actions/workflows/test_lint.yaml)


---

## What is Allen Cell ML Segmenter
A deep learning-based segmentation Napari plugin to curate datasets, train your own model (UNET), and run inference on 2D and 3D cell data. 


##  News

 - **[2024.09.20]** Initial release.

See the [changelog](https://github.com/AllenCell/allencell-ml-segmenter/tree/main/CHANGELOG.md) for more details.

## Installation

### System Requirements

We currently support `Windows`, `MacOS`, and `Linux` operating systems. The minimum system requirements are:

- 8GB of RAM
- 8 CPU Cores
- 1 NVIDIA GPU with 8GB of VRAM (optional)

**NOTE:** If you plan to use the plugin _without_ a GPU, training will default to using your CPU and will be significantly slower. A GPU is highly recommended for training models. Depending on how large your images are---2D vs 3D, resolution, model size---running inference may also be slow without a GPU.

### Pre-Installation

__STEP 1.__ Before installing the plugin, please make sure you have the following installed:

- Python 3.10 or later

__New to `Python`?__ We recommend installing `Python 3.10` through the official [`Python` website](https://www.python.org/downloads/). This will include the `pip` package manager, which is required to install the plugin.

If you are unsure if you have Python installed or which version you may have, you can check by running the following command in your terminal or powershell:

```bash
# Check version of python
python --version

# If the above does not work, try this one
python3 --version

# Specifically check for Python 3.10
python3.10 --version
```



__STEP 2.__ Next we will create a new `Python` environment to install the plugin. This will help avoid conflicts with other packages you may have installed and create an isolated environment for the plugin to live. In general, it is good practice to choose a name for your environment that is related to either the project you are working on or the software you are installing. In this case, we use `allen-ml-segmenter`.

Navigate to where you want to create a new environment (_Example._ `Documents`), run the following command in your terminal or powershell:

```bash
# Create a new environment
python3.10 -m venv allen-ml-segmenter

# Activate the environment
source allen-ml-segmenter/bin/activate
```




### Install the Plugin

To install the latest version of the plugin:
```bash
pip install allencell-ml-segmenter
```



## Models
| Model    | Model Name            | Available in Plugin | Model Size (MB)  | Supported Magnifications| 
|----------|-----------------------|----------------------------------|----------------------------------------|:-------------------------:|
| Megaseg-S  | `megaseg_small`      | ✅        | 10MB      |       10X,20X         |          
| Megaseg-M  | `megaseg_medium`     | Coming soon!       |  50MB     |       10X,20X        |           
| Megaseg-L  | `megaseg_large`      | ✅        | 192MB       |       10X,20X,67X        |  


## License

Distributed under the terms of the [Allen Institute Software License] license.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[@napari]: https://github.com/napari
[Allen Institute Software License]: /LICENSE
[file an issue]: https://github.com/AllenCell/allencell-ml-segmenter/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
