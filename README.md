# allencell-ml-segmenter

[![License BSD-3](https://img.shields.io/pypi/l/allencell-ml-segmenter.svg?color=green)](https://github.com/AllenCell/allencell-ml-segmenter/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/allencell-ml-segmenter.svg?color=green)](https://pypi.org/project/allencell-ml-segmenter)
[![Python Version](https://img.shields.io/pypi/pyversions/allencell-ml-segmenter.svg?color=green)](https://python.org)
[![Tests](https://github.com/AllenCell/allencell-ml-segmenter/actions/workflows/test_lint.yaml/badge.svg)](https://github.com/AllenCell/allencell-ml-segmenter/actions/workflows/test_lint.yaml)
[![codecov](https://codecov.io/gh/AllenCell/allencell-ml-segmenter/branch/main/graph/badge.svg?token=E976SiYFP6)](https://codecov.io/gh/AllenCell/allencell-ml-segmenter)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/allencell-ml-segmenter)](https://napari-hub.org/plugins/allencell-ml-segmenter)

A plugin to leverage ML segmentation in napari.

---

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Dev setup

_Note on Mac OS, use `gmake` instead of `make`_

First, set up a venv and installing project dependencies into it, use `make install-dev`.

More useful dev tasks:

- `make clean` [clean the venv and other build/test artifacts]
- `make test` [Run unit tests]
- `make lint` [Find lint errors in source code]
- `make format` [Format source code]

## Current steps for integrating plugin with cyto-del:

- clone plugin (use main - my stuff is merged)
- git checkout bugfix/allencell-ml-segmenter
- clone cyto-dl as sibling to plugin repo
- cd into cyto-dl and download data: python scripts/download_test_data.py
- copy `/cyto-dl/data` to `allencell-ml-segmenter/`.  The input image paths are references in train.csv as relative paths, and so must be resolved at PYTHON_PATH (in the plugin repo).
- from the plugin repo, `gmake install`
- activate new venv (happens autmatically i think)
- `pip install .`
- `pip install PyQt5`
- `python -m pip install -e ../cyto-dl/`
- `touch cyto-dl/configs/__init__.py` (necessary to reference `segmenter.yaml` in the cyto-dl repo)
- [DEPRACATED] update hardcoded paths in TrainingService for your system
- update the paths stored in `constants.py` to work on your system
- `napari`
- select training view, hit training button. It should run to completion.

## Releasing

Release a new version and publishing to Pypi is based on GitHub Actions workflows.

**To create a standard release (new major, minor, patch, or dev version):**

- From repository homepage, go to `actions > Bump version and PR`
- Enter which semantic version component you want to bump for this release
- Run the workflow
- A GitHub runner will then bump the version, and PR the change back to main
- Merge the PR
- From repository homepage, go to `actions > Publish tagged version`
- Enter the tag that was created for this version (e.g. `v0.2.3`) 
- Enter the target index for publishing
- Run the workflow

**To create a hotfix (post version of existing release):**

- Branch off of the version tag for the existing release
- Make necessary changes, commit and push
- From repository homepage, go to `actions > Create or increment a hotfix/post tag on a branch`
- Enter the name of the branch you created in the first step
- Run the workflow
- This will bump the version of the latest commit on the branch and tag it, pushing the changes back to the hotfix branch
- You can make more changes and increment the hotfix tag again if necessary
- When you want to publish the hotfix...
- From repository homepage, go to `actions > Publish tagged version`
- Enter the hotfix tag that was created for this version (e.g. `v0.2.3.post0`) 
- Enter the target index for publishing
- Run the workflow

**Note**: There are restrictions on which component you can bump depending on the current version.
If the current version has a dev component (e.g. `1.0.1.dev0`), you must bump the dev component or the patch component
(bumping patch will finalize that patch without any dev version `1.0.1.dev0` -> `1.0.1`). Attempting to bump
major or minor versions will cause the workflow to fail. If the current version has a post component (e.g. `1.0.1.post0`),
you can only bump the post version.

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
