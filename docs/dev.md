## Dev setup

_Note on Mac OS, use `gmake` instead of `make`_

First, set up a venv and installing project dependencies into it, use `make install-dev`.

More useful dev tasks:

- `make clean` [clean the venv and other build/test artifacts]
- `make test` [Run unit tests]
- `make lint` [Find lint errors in source code]
- `make format` [Format source code]

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