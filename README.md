# allencell-ml-segmenter

[![License BSD-3](https://img.shields.io/pypi/l/allencell-ml-segmenter.svg?color=green)](https://github.com/AllenCell/allencell-ml-segmenter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/allencell-ml-segmenter.svg?color=green)](https://pypi.org/project/allencell-ml-segmenter)
[![Python Version](https://img.shields.io/pypi/pyversions/allencell-ml-segmenter.svg?color=green)](https://python.org)
[![Tests](https://github.com/AllenCell/allencell-ml-segmenter/actions/workflows/test_lint.yaml/badge.svg)](https://github.com/AllenCell/allencell-ml-segmenter/actions/workflows/test_lint.yaml)
[![codecov](https://codecov.io/gh/AllenCell/allencell-ml-segmenter/branch/main/graph/badge.svg?token=E976SiYFP6)](https://codecov.io/gh/AllenCell/allencell-ml-segmenter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/allencell-ml-segmenter)](https://napari-hub.org/plugins/allencell-ml-segmenter)

A plugin to leverage ML segmentation in napari.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Dev setup

*Note on Mac OS, use `gmake` instead of `make`*

First, set up a venv and installing project dependencies into it, use `make install-dev`.

More useful dev tasks:
* `make clean` [clean the venv and other build/test artifacts]
* `make test` [Run unit tests]
* `make lint` [Find lint errors in source code]
* `make format` [Format source code]

## Current steps for integrating plugin with cyto-del:

* clone plugin (use main - my stuff is merged)
* git checkout bugfix/allencell-ml-segmenter
* clone cyto-dl as sibling to plugin repo
* cd into cyto-dl and download data: python scripts/download_test_data.py
* `cp -r ../cyto-dl/data ./data` (input image paths currently must be under * * plugin working dir)
* `gmake install`
* activate new venv (happens autmatically i think)
* `pip install PyQt5`
* `python -m pip install -e ../cyto-dl/`
* update hardcoded paths in TrainingService for your system
* `napari`
* select training view, hit training button.  It should run to completion.

## Releasing

Release a new version and publishing to Pypi is based on a Github Actions workflow.  The steps are:

* Increment the current version using `bumpversion`.  This can be done directly, or using the convenience `make` tasks (eg. `make bumpversion-patch`)
* Assuming that `main` is up to date with the changes that you intend to release, make a pr from `main` into `release`.  Upon getting required approvals and merging, the new version will be published to Pypi and released on Github.  
    * Branch protections apply - currently, merging into `release` is restricted to project maintainers.

## Installation

You can install `allencell-ml-segmenter` via [pip]:

    pip install allencell-ml-segmenter



To install latest development version :

    pip install git+https://github.com/AllenCell/allencell-ml-segmenter.git


## Contributing

Contributions are very welcome. Tests can be run with pytest, or `make test`, please ensure
the coverage at least stays the same before you submit a pull request.

To check coverage, run pytest with the '--cov' flag:
    `pytest --cov=allencell_ml_segmenter`
or use `make test-cov`.

## License

Distributed under the terms of the [BSD-3] license,
"allencell-ml-segmenter" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/AllenCell/allencell-ml-segmenter/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
