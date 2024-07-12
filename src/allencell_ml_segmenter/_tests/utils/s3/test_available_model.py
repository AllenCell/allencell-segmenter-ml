from pathlib import Path
import responses
import pytest
from pytest import MonkeyPatch

import allencell_ml_segmenter
from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.utils.s3.s3_available_model import AvailableModel
from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)
from allencell_ml_segmenter.utils.zip_file.fake_zip_file_manager import (
    FakeZipFileManager,
)


@responses.activate
def test_download_model_and_unzip_sucessful_request() -> None:
    # fake data
    fake_content: str = "fake_content abcde"
    fake_url: str = "http://test.com/abc"
    fake_model_file_name: str = "some_new_model.zip"
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{fake_url}/{fake_model_file_name}",
            "body": fake_content,
            "status": 200,
            "content_type": "application/zip",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )
    # Test path to save zip to
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
    )
    fake_zip_file_manager: FakeZipFileManager = FakeZipFileManager()
    available_model: AvailableModel = AvailableModel(
        fake_model_file_name, fake_url, test_path, fake_zip_file_manager
    )

    # Act
    available_model.download_model_and_unzip()

    # Assert
    assert fake_zip_file_manager.written_zip_files[
        test_path / fake_model_file_name
    ] == bytes(fake_content, "utf-8")
    assert (
        test_path / fake_model_file_name
        in fake_zip_file_manager.unzipped_files
    )


@responses.activate
def test_download_model_and_unzip_bad_request() -> None:
    # ARRANGE
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
        / "test_zip.zip"
    )
    fake_url: str = "https://testurl.com/test_url"
    fake_zip_file_manager: FakeZipFileManager = FakeZipFileManager()
    available_model: AvailableModel = AvailableModel(
        "abc", fake_url, test_path, fake_zip_file_manager
    )
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{fake_url}/abc",
            "body": '{"error": "error_reason"}',
            "status": 400,
            "content_type": "application/zip",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )

    # Act/Assert
    with pytest.raises(S3RequestException):
        available_model.download_model_and_unzip()


@responses.activate
def test_download_model_and_unzip_model_exists_overwrite(
    monkeypatch: MonkeyPatch,
) -> None:
    # standard way to deal with modal dialogs: https://pytest-qt.readthedocs.io/en/latest/note_dialogs.html
    monkeypatch.setattr(DialogBox, "exec", lambda *args: 0)
    monkeypatch.setattr(
        DialogBox, "get_selection", lambda x: True
    )  # mimic user clicks ovewrite
    # fake data
    fake_content: str = "fake_content abcde"
    fake_url: str = "http://test.com/abc"
    # this is an existing "model" name in the test files directory
    fake_model_file_name: str = "test_zip.zip"
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{fake_url}/{fake_model_file_name}",
            "body": fake_content,
            "status": 200,
            "content_type": "application/zip",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )
    # Test path to save zip to
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
    )
    fake_zip_file_manager: FakeZipFileManager = FakeZipFileManager()
    available_model: AvailableModel = AvailableModel(
        fake_model_file_name, fake_url, test_path, fake_zip_file_manager
    )

    # Act
    available_model.download_model_and_unzip()

    # Assert
    assert fake_zip_file_manager.written_zip_files[
        test_path / fake_model_file_name
    ] == bytes(fake_content, "utf-8")
    assert (
        test_path / fake_model_file_name
        in fake_zip_file_manager.unzipped_files
    )


@responses.activate
def test_download_model_and_unzip_model_exists_do_not_overwrite(
    monkeypatch: MonkeyPatch,
) -> None:
    # standard way to deal with modal dialogs: https://pytest-qt.readthedocs.io/en/latest/note_dialogs.html
    monkeypatch.setattr(DialogBox, "exec", lambda *args: 0)
    monkeypatch.setattr(
        DialogBox, "get_selection", lambda x: False
    )  # mimic user clicking dont overwrite
    # fake data
    fake_content: str = "fake_content abcde"
    fake_url: str = "http://test.com/abc"
    # this is an existing "model" name in the test files directory
    fake_model_file_name: str = "test_zip.zip"
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{fake_url}/{fake_model_file_name}",
            "body": fake_content,
            "status": 200,
            "content_type": "application/zip",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )
    # Test path to save zip to
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
    )
    fake_zip_file_manager: FakeZipFileManager = FakeZipFileManager()
    available_model: AvailableModel = AvailableModel(
        fake_model_file_name, fake_url, test_path, fake_zip_file_manager
    )

    # Act
    available_model.download_model_and_unzip()

    # Assert
    assert len(fake_zip_file_manager.written_zip_files) == 0
    assert len(fake_zip_file_manager.unzipped_files) == 0
