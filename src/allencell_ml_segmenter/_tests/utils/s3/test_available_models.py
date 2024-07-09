from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import allencell_ml_segmenter
from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.s3.s3_available_models import AvailableModels
import responses
import pytest

from allencell_ml_segmenter.utils.s3.s3_request_exception import S3RequestException
from allencell_ml_segmenter.utils.zip_file.fake_zip_file_manager import FakeZipFileManager


@responses.activate
def test_download_model_and_unzip_sucessful_request():
    # Test path to save zip to
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
    )
    # fake data
    fake_url: str = "https://testurl.com/test_url"
    fake_content: str = "fake_content abcde"
    fake_model_file_name: str = "test_model.zip"
    fake_zip_file_manager = FakeZipFileManager.global_instance()
    available_model: AvailableModels = AvailableModels(fake_model_file_name, fake_url, fake_zip_file_manager)
    responses.add(**{
        'method': responses.GET,
        'url': fake_url,
        'body': '{"content": ' + fake_content + '}',
        'status': 200,
        'content_type': 'application/zip',
        'adding_headers': {'X-Foo': 'Bar'}
    })

    # Act
    available_model.download_model_and_unzip(test_path)

    # Assert
    assert fake_zip_file_manager.written_zip_files[test_path / fake_model_file_name] == fake_content
    assert test_path / fake_model_file_name in fake_zip_file_manager.unzipped_files

@responses.activate
def test_download_model_and_unzip_bad_request():
    # ARRANGE
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
        / "test_zip.zip"
    )
    fake_url: str = "https://testurl.com/test_url"
    fake_zip_file_manager = FakeZipFileManager.global_instance()
    available_model: AvailableModels = AvailableModels("abc", fake_url, fake_zip_file_manager)
    responses.add(**{
        'method': responses.GET,
        'url': fake_url,
        'body': '{"error": "error_reason"}',
        'status': 400,
        'content_type': 'application/zip',
        'adding_headers': {'X-Foo': 'Bar'}
    })

    # Act/Assert
    with pytest.raises(S3RequestException):
        available_model.download_model_and_unzip(test_path)
