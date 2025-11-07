# Allencell-segmenter-ml

[![Test and lint](https://github.com/AllenCell/allencell-segmenter-ml/actions/workflows/test_lint.yaml/badge.svg?branch=main&event=push)](https://github.com/AllenCell/allencell-segmenter-ml/actions/workflows/test_lint_pr.yaml)


## What is Allen Cell Segmenter ML
A napari plugin for deep-learning based segmentation of cellular structures.

![SegmenterML-plugin_fig1_output.png](docs%2Fuser_docs%2Fimages%2FSegmenterML-plugin_fig1_output.png)

- **Available at no cost** — available on PyPI
- **User-friendly** — leverage napari as a fast 3D viewer with interactive plugin interface
- **Beginner-friendly** — new to machine learning? This plugin simplifies the application of machine learning in the segmentation process through the 3 main modules:
  - **Curation**: curate training datasets
  - **Training**: iteratively train custom segmentation model(s) (UNET) to target cellular structure with wide morphological variability
  - **Prediction & Thresholding**: generate segmentation prediction on 2D and 3D cell image data


##  📰 News

 - **[2024.09.24]** :tada: Initial release of the plugin and Megaseg models!
 - **[2024.05.29]** :tada: v1.0.0 Released on PyPi


## User Documentation
[See our full user documentation on our github pages site.](https://allencell.github.io/allencell-segmenter-ml/index.html)


## 🛠️ Installation

### System and Data Requirements

[Please click here to check out our latest System and Data requirements.](https://allencell.github.io/allencell-segmenter-ml/1_Get-started/1_prerequisites.html)


### Installation Steps
[Please click here for our latest installation steps.](https://allencell.github.io/allencell-segmenter-ml/1_Get-started/2_installation.html)


## Models
[More information about the pre-trained models we provide with our plugin, and citation information, can be found here.](https://allencell.github.io/allencell-segmenter-ml/1_Get-started/4_pretrained-models.html)

## License

Distributed under the terms of the [Allen Institute Software License].

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[@napari]: https://github.com/napari
[Allen Institute Software License]: https://github.com/AllenCell/allencell-segmenter-ml/blob/main/LICENSE
[file an issue]: https://github.com/AllenCell/allencell-ml-segmenter/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[PyTorch]: https://pytorch.org/get-started/locally/
