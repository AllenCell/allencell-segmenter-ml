# this file is intended to be called by a github workflow (.github/workflows/publish_to_pypi.yaml)
# it makes decisions based on the current version and the component specified for bumping,
# which the workflow cannot do
import subprocess
import sys
import toml  # type: ignore


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError("No component specified for bumping version")

    component: str = sys.argv[1].lower()
    valid_options: set[str] = {"major", "minor", "patch", "rc", "post"}

    if component not in valid_options:
        raise ValueError(f"Component must be one of {valid_options}")

    version: str = toml.load("pyproject.toml")["project"]["version"]
    version_components: list[str] = version.split(".")

    update_output: subprocess.CompletedProcess
    if "rc" in version_components[-1]:
        if component == "rc":
            # increment the rc tag (e.g. 1.0.0rc0 -> 1.0.0rc1)
            update_output = subprocess.run(
                ["bumpver", "update", "--tag-num", "-n"]
            )
        elif component == "patch":
            # finalize the patch by removing rc tag (e.g. 1.0.0rc1 -> 1.0.0)
            update_output = subprocess.run(
                ["bumpver", "update", "--tag=final", "-n"]
            )
        else:
            raise ValueError(
                "Cannot update major or minor version while rc version is current"
            )
    elif "post" in version_components[-1]:
        if component == "post":
            update_output = subprocess.run(
                ["bumpver", "update", "--tag-num", "-n"]
            )
        else:
            raise ValueError("Cannot change post version to standard version")
    else:
        if component == "rc":
            # increment patch and begin at rc0 (e.g. 1.0.0 -> 1.0.1rc0)
            update_output = subprocess.run(
                ["bumpver", "update", "--patch", "--tag=rc", "-n"]
            )
        elif component == "post":
            update_output = subprocess.run(
                ["bumpver", "update", "--tag=post", "-n"]
            )
        else:
            update_output = subprocess.run(
                ["bumpver", "update", f"--{component}", "-n"]
            )

    if update_output.returncode != 0:
        raise RuntimeError(
            f"bumpver exited with code {update_output.returncode}"
        )


if __name__ == "__main__":
    main()
