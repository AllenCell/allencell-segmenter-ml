# this file is intended to be called by a github workflow (.github/workflows/publish_to_pypi.yaml)
# it encapsulates logic that is too complex to be expressed in workflow syntax
import subprocess
import sys
from typing import Set, List

def main():
    if len(sys.argv) < 2:
        raise ValueError("No component specified for bumping version")

    component: str = sys.argv[1].lower()
    valid_options: Set[str] = {"major", "minor", "patch", "dev"}

    if component not in valid_options:
        raise ValueError(f"Component must be one of {valid_options}")

    show_output: subprocess.CompletedProcess = subprocess.run(["bumpver", "show", "-n"], capture_output=True)
    if show_output.returncode != 0:
        raise RuntimeError(f"bumpver exited with code {show_output.returncode}")

    # brittle, may break if 'bumpver show -n' output changes, so pinning bumpver dependency in
    # pyproject.toml:project.optional_dependencies.build_and_publish
    current_version: str = show_output.stdout.decode().split("\n")[0].split(": ")[-1].strip()
    version_components: List[str] = current_version.split(".")

    update_output: subprocess.CompletedProcess = None
    # 4 components means we currently have a dev version
    if len(version_components) == 4:
        if component == "dev":
            # increment the dev tag (e.g. 1.0.0.dev0 -> 1.0.0.dev1)
            update_output = subprocess.run(["bumpver", "update", "--tag-num"])
        elif component == "patch":
            # finalize the patch by removing dev tag (e.g. 1.0.0.dev1 -> 1.0.0)
            update_output = subprocess.run(["bumpver", "update", "--tag=final"])
        else:
            raise ValueError("Cannot update major or minor version while dev version is current")

    elif len(version_components) == 3:
        if component == "dev":
            # increment patch and begin at dev0 (e.g. 1.0.0 -> 1.0.1.dev0)
            update_output = subprocess.run(["bumpver", "update", "--patch", "--tag=dev"])
        else:
            update_output = subprocess.run(["bumpver", "update", f"--{component}"])

    else:
        raise ValueError(f"Unknown version format: {current_version}. Expected MAJOR.MINOR.PATCH[.PYTAGNUM]")
    
    if update_output.returncode != 0:
        raise RuntimeError(f"bumpver exited with code {update_output.returncode}")
    

if __name__ == "__main__":
    main()

"""
TESTING:
- add and commit any changes (keep track of this commit hash)
- bumpver update --set-version 1.0.0

- python publish_bumpver_handler.py
  - expect: ValueError

- python publish_bumpver_handler.py fake
  - expect: ValueError

- python publish_bumpver_handler.py major
  - expect: version updated to 2.0.0

- python publish_bumpver_handler.py minor
  - expect: version updated to 2.1.0

- python publish_bumpver_handler.py patch
  - expect: version updated to 2.1.1

- python publish_bumpver_handler.py dev
  - expect: version updated to 2.1.2.dev0

- python publish_bumpver_handler.py dev
  - expect: version updated to 2.1.2.dev1

- python publish_bumpver_handler.py major
  - expect: ValueError

- python publish_bumpver_handler.py minor
  - expect: ValueError

- python publish_bumpver_handler.py patch
  - expect: version updated to 2.1.2

- git reset --hard {hash of the commit made at the beginning}
"""