from pathlib import Path

import requests

from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.s3.s3_request_exception import S3RequestException
from allencell_ml_segmenter.utils.zip_file import IZipFileManager, ZipFileManager


class AvailableModels:
    def __init__(self, model_file_name: str, bucket_endpoint: str, zip_file_manager: IZipFileManager = ZipFileManager.global_instance()):
        self._name = model_file_name
        self._object_url: str = f"{bucket_endpoint}/{model_file_name}"
        self._zipfile = zip_file_manager

    def download_model_and_unzip(self, path: Path) -> None:
        response = requests.get(self._object_url)

        if response.status_code == 200:
            # write a zipfile to path
            self._zipfile.write_zip_file(path / self._name, response.content)
            # unzip zipfile containing model and license file, and delete original zip file
            self._zipfile.unzip_zipped_file_and_delete_zip(path / self._name)
        else:
            raise S3RequestException(f"Could not download model named {self._name} from {self._object_url}. Failed with status code {response.status_code}")


