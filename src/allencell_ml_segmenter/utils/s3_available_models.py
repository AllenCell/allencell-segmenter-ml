from pathlib import Path

import requests

from allencell_ml_segmenter.utils.file_utils import FileUtils
from allencell_ml_segmenter.utils.s3_model_downloader import S3RequestException


class AvailableModels:
    def __init__(self, model_file_name: str, bucket_endpoint: str):
        self._name = model_file_name
        self._object_url: str = f"{bucket_endpoint}/{model_file_name}"

    def download_model_and_unzip(self, path: Path) -> None:
        response = requests.get(self._object_url)

        if response.status_code == 200:
            FileUtils.write_to_file(path, response.content)
        else:
            raise S3RequestException(f"Could not download model named {self._name} from {self._object_url}. Failed with status code {response.status_code}")
        FileUtils.unzip_zipped_file_and_delete_zip(Path(path) / self._name)
