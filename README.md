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

We recommend using PDM for dev work.

### Getting started
1. [install PDM](https://pdm-project.org/en/latest/#installation) if not already installed (on macOS, I prefer the `brew install pdm` method).
2. clone this repository.
3. in the root of the cloned repository, run `pdm install -G dev`. This will create a `.venv` virtual environment and install all dev dependencies there. It will also automatically install this repo as an editable package.
4. activate the virtual environment (either through your IDE or with something like `source .venv/bin/activate` depending on your OS)
5. run `napari` in your shell

Congrats! You now have a working editable installation of segmenter ML--you can develop and see your changes live now by re-running `napari`.

### Adding dependencies
If you need to add a dependency to the project, do so by running `pdm add <dep>`. This will automatically update the lock file and `pyproject.toml`. Remember to commit both `pdm.lock` and `pyproject.toml`. You **do not** need to commit `.pdm-python` or any other PDM artifacts.

If you want to add a dev-only dependency, use `pdm add -dG dev <dep>`.

### Editable installs for other packages
I found this useful when I had a branch on `cyto-dl` I wanted to test segmenter with:

`pdm add -e git+https://github.com/AllenCellModeling/cyto-dl.git@make-req-optional#egg=cyto-dl --dev`

Remember if you are adding an editable dependency, you **should not** commit the changed `pdm.lock` or `pyproject.toml`, as this is only a temporary solution for testing.

## Releasing

Release a new version and publishing to Pypi is based on a Github Actions workflow. The steps are:

- From repository homepage, go to `actions > Bump version and publish to PyPI`
- Enter which semantic version component you want to bump for this release
- Run the workflow

A GitHub runner will then bump the version, build and release to PyPI, and push the version changes back to main.

**Note**: There are restrictions on which component you can bump depending on the current version.
If the current version has a dev component (e.g. `1.0.1.dev0`), you must bump the dev component or the patch component
(bumping patch will finalize that patch without any dev version `1.0.1.dev0` -> `1.0.1`). Attempting to bump
major or minor versions will cause the workflow to fail

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
