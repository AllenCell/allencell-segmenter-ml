import sys
import toml
import os
import shutil
import allencell_ml_segmenter.scripts.bumpver_handler as bumpver_handler
from pathlib import Path

DEFAULT_START_VERSION: str = "0.0.1"


# tests in this file must call navigate_to_test_dir_and_reset_version at the beginning and navigate_back_to_root_dir at the end
def navigate_to_test_dir_and_reset_version() -> None:
    os.chdir(Path(__file__).parent)
    shutil.copyfile("unmodified_pyproject.toml", "pyproject.toml")


def navigate_back_to_root_dir() -> None:
    os.chdir("../../../..")


def assert_curr_version_number(version: str) -> None:
    curr_version: str = toml.load("pyproject.toml")["project"]["version"]
    assert curr_version == version


def test_bump_patch() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT
    sys.argv = ["bumpver_handler.py", "patch"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.2")
    # ACT
    sys.argv = ["bumpver_handler.py", "patch"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.3")
    navigate_back_to_root_dir()

def test_bump_minor() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT
    sys.argv = ["bumpver_handler.py", "minor"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.1.0")
    # ACT
    sys.argv = ["bumpver_handler.py", "minor"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.2.0")
    navigate_back_to_root_dir()

def test_bump_major() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT
    sys.argv = ["bumpver_handler.py", "major"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("1.0.0")
    # ACT
    sys.argv = ["bumpver_handler.py", "major"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("2.0.0")
    navigate_back_to_root_dir()

def test_bump_dev() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new patch version with .dev0 tag)
    sys.argv = ["bumpver_handler.py", "dev"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.2.dev0")
    # ACT
    sys.argv = ["bumpver_handler.py", "dev"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.2.dev1")
    # ACT (finalize the new patch version)
    sys.argv = ["bumpver_handler.py", "patch"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.2")
    navigate_back_to_root_dir()

def test_bump_post() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new post version with .post1 tag)
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.1.post1")
    # ACT
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.1.post2")
    navigate_back_to_root_dir()