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
    valid_options: set[str] = {"major", "minor", "patch", "dev", "post"}

    if component not in valid_options:
        raise ValueError(f"Component must be one of {valid_options}")

    version: str = toml.load("pyproject.toml")["project"]["version"]
    version_components: list[str] = version.split(".")

    update_output: subprocess.CompletedProcess
    # 4 components means we currently have a dev or post version
    if len(version_components) == 4 and version_components[-1].startswith(
        "dev"
    ):
        if component == "dev":
            # increment the dev tag (e.g. 1.0.0.dev0 -> 1.0.0.dev1)
            update_output = subprocess.run(
                ["bumpver", "update", "--tag-num", "-n"]
            )
        elif component == "patch":
            # finalize the patch by removing dev tag (e.g. 1.0.0.dev1 -> 1.0.0)
            update_output = subprocess.run(
                ["bumpver", "update", "--tag=final", "-n"]
            )
        else:
            raise ValueError(
                "Cannot update major or minor version while dev version is current"
            )
    elif len(version_components) == 4:  # current version must be post
        if component == "post":
            update_output = subprocess.run(
                ["bumpver", "update", "--tag-num", "-n"]
            )
        else:
            raise ValueError("Cannot change post version to standard version")
    elif len(version_components) == 3:
        if component == "dev":
            # increment patch and begin at dev0 (e.g. 1.0.0 -> 1.0.1.dev0)
            update_output = subprocess.run(
                ["bumpver", "update", "--patch", "--tag=dev", "-n"]
            )
        elif component == "post":
            update_output = subprocess.run(
                ["bumpver", "update", "--tag=post", "-n"]
            )
        else:
            update_output = subprocess.run(
                ["bumpver", "update", f"--{component}", "-n"]
            )

    else:
        raise ValueError(
            f"Unknown version format: {version}. Expected MAJOR.MINOR.PATCH[.PYTAGNUM]"
        )

    if update_output.returncode != 0:
        raise RuntimeError(
            f"bumpver exited with code {update_output.returncode}"
        )


if __name__ == "__main__":
    main()
