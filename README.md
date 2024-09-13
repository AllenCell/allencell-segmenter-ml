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
A deep learning-based segmentation Napari plugin to train and run inference on 2D and 3D cell structures.


## :sparkles: News

 - **[2024.09.20]** Initial release.

See the [changelog](https://github.com/AllenCell/allencell-ml-segmenter/tree/main/CHANGELOG.md) for more details.

## Installation

Hiera requires a reasonably recent version of [torch](https://pytorch.org/get-started/locally/).
After that, you can install hiera through [pip](https://pypi.org/project/hiera-transformer/):
```bash
pip install 
```

## Models
| Model    | Model Name            | Pretrained Models<br>(IN-1K MAE) | Finetuned Models<br>(IN-1K Supervised) | IN-1K<br>Top-1 (%) | A100 fp16<br>Speed (im/s) |
|----------|-----------------------|----------------------------------|----------------------------------------|:------------------:|:-------------------------:|
| Hiera-T  | `hiera_tiny_224`      | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth)        | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth)       |       82.8         |            2758           |
| Hiera-S  | `hiera_small_224`     | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth)       | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth)      |       83.8         |            2211           |
| Hiera-B  | `hiera_base_224`      | [mae_in1k](https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth)        | [mae_in1k_ft_in1k](https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth)       |       84.5         |            1556           |


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
