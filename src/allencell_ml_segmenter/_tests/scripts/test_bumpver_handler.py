import sys
import toml
import os
import shutil
import allencell_ml_segmenter.scripts.bumpver_handler as bumpver_handler
from pathlib import Path

"""
The tests in this file are a bit unconventional. Continue reading for an explanation.

The purpose of bumpver_handler.py is to handle the logic around the args we need to pass
bumpver--the args vary based on which version component we want to bump AND the current version.
This logic is too complex for GH workflows to handle, so we created bumpver_handler.py, which
is called by GH workflows.

We want to verify that the args we pass to bumpver actually make the version updates we expect.
This is tricky to test because bumpver is a tool that reads version state from the repo files and
writes updates to files in the repo, but we don't actually want to change our repo's version over the
course of testing.

The solution here is that we create a 'fake' repo (meaning a fake pyproject.toml) in the _tests/scripts
directory. Then when it comes time to run the bumpver_handler tests, we change our working directory
to _tests/scripts, which tricks bumpver into thinking that our 'fake' pyproject.toml is the source 
of truth.

So, the tests work as follows:
  - change working directory to _tests/scripts
  - copy unmodified_pyproject.toml (a frozen pyproject file that won't be touched by bumpver) to
    pyproject.toml, which resets the version to 0.0.1
  - run bumpver_handler.py, changing sys.argv to modify the args we provide it
  - read pyproject.toml to make sure the version updated as we expected
  - change working directory back to the root of the repo

The first and last steps are accomplished with navigate_to_test_dir_and_reset_version
and navigate_back_to_root_dir, respectively
"""
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


def test_bump_rc() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new patch version with .rc0 tag)
    sys.argv = ["bumpver_handler.py", "rc"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.2rc0")
    # ACT
    sys.argv = ["bumpver_handler.py", "rc"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.2rc1")
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
    # ACT (create new post version with .post0 tag)
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.1post0")
    # ACT
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT
    assert_curr_version_number("0.0.1post1")
    navigate_back_to_root_dir()


def test_bump_minor_fails_when_rc_current() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new rc version with .rc0 tag)
    sys.argv = ["bumpver_handler.py", "rc"]
    bumpver_handler.main()
    # ASSERT (sanity check)
    assert_curr_version_number("0.0.2rc0")

    got_expected_exception: bool = False
    try:
        sys.argv = ["bumpver_handler.py", "minor"]
        bumpver_handler.main()
    except ValueError as e:
        got_expected_exception = True
    except:
        pass

    # ASSERT
    assert got_expected_exception
    navigate_back_to_root_dir()


def test_bump_major_fails_when_rc_current() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new rc version with .rc0 tag)
    sys.argv = ["bumpver_handler.py", "rc"]
    bumpver_handler.main()
    # ASSERT (sanity check)
    assert_curr_version_number("0.0.2rc0")

    got_expected_exception: bool = False
    try:
        sys.argv = ["bumpver_handler.py", "major"]
        bumpver_handler.main()
    except ValueError as e:
        got_expected_exception = True
    except:
        pass

    # ASSERT
    assert got_expected_exception
    navigate_back_to_root_dir()


def test_bump_post_fails_when_rc_current() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new rc version with .rc0 tag)
    sys.argv = ["bumpver_handler.py", "rc"]
    bumpver_handler.main()
    # ASSERT (sanity check)
    assert_curr_version_number("0.0.2rc0")

    got_expected_exception: bool = False
    try:
        sys.argv = ["bumpver_handler.py", "post"]
        bumpver_handler.main()
    except ValueError as e:
        got_expected_exception = True
    except:
        pass

    # ASSERT
    assert got_expected_exception
    navigate_back_to_root_dir()


def test_bump_minor_fails_when_post_current() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new post version with .post0 tag)
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT (sanity check)
    assert_curr_version_number("0.0.1post0")

    got_expected_exception: bool = False
    try:
        sys.argv = ["bumpver_handler.py", "minor"]
        bumpver_handler.main()
    except ValueError as e:
        got_expected_exception = True
    except:
        pass

    # ASSERT
    assert got_expected_exception
    navigate_back_to_root_dir()


def test_bump_major_fails_when_post_current() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new post version with .post0 tag)
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT (sanity check)
    assert_curr_version_number("0.0.1post0")

    got_expected_exception: bool = False
    try:
        sys.argv = ["bumpver_handler.py", "major"]
        bumpver_handler.main()
    except ValueError as e:
        got_expected_exception = True
    except:
        pass

    # ASSERT
    assert got_expected_exception
    navigate_back_to_root_dir()


def test_bump_patch_fails_when_post_current() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new post version with .post0 tag)
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT (sanity check)
    assert_curr_version_number("0.0.1post0")

    got_expected_exception: bool = False
    try:
        sys.argv = ["bumpver_handler.py", "patch"]
        bumpver_handler.main()
    except ValueError as e:
        got_expected_exception = True
    except:
        pass

    # ASSERT
    assert got_expected_exception
    navigate_back_to_root_dir()


def test_bump_rc_fails_when_post_current() -> None:
    navigate_to_test_dir_and_reset_version()
    # ASSERT (sanity check)
    assert_curr_version_number(DEFAULT_START_VERSION)
    # ACT (create new post version with .post0 tag)
    sys.argv = ["bumpver_handler.py", "post"]
    bumpver_handler.main()
    # ASSERT (sanity check)
    assert_curr_version_number("0.0.1post0")

    got_expected_exception: bool = False
    try:
        sys.argv = ["bumpver_handler.py", "rc"]
        bumpver_handler.main()
    except ValueError as e:
        got_expected_exception = True
    except:
        pass

    # ASSERT
    assert got_expected_exception
    # For version control niceness, we should have this at the end of the
    # last test in this file
    navigate_to_test_dir_and_reset_version()
    navigate_back_to_root_dir()
