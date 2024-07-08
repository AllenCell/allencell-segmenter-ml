import zipfile
from enum import Enum
from pathlib import Path
from typing import Optional, Set
from xml.etree.ElementTree import ElementTree
from zipfile import ZipFile

import boto3
import requests

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel


ENABLE_MODEL_DOWNLOADS = True
BUCKET_ENDPOINT = ""
STG_BUCKET_ENDPOINT = ""

class S3RequestException(Exception):
    pass

class S3ModelDownloader:
    def __init__(self, staging=False):
        self._bucket_endpoint = STG_BUCKET_ENDPOINT if staging else BUCKET_ENDPOINT

    def get_available_models(self) -> list[str]:
        """
        Get a list of the filenames of available models on s3.

        Uses a http request to get all available models on s3. This function assumes all models will be in .zip files
        (and ignores all files that are not zip files). Any errors with the request throws a S3RequestException.
        """
        # request parameter to list all objects in bucket
        req_params: str = "list-type=2"

        response = requests.get(f"{self._bucket_endpoint}?{req_params}")

        if response.status_code == 200:
            # parse XML response
            xml_root = ElementTree.fromstring(response.content)
            # extract s3 object names
            available: list[str] = [xml_content.find('Key').text for xml_content in xml_root.findall('.//Contents')]
            # ignore all files that are not zip (like readme/other top level files that aren't models),
            # and only grab name of the file ignoring extension
            available = [obj.split(".")[0] for obj in available if not obj.endswith('.zip')]
            return available
        else:
            # handle request errors
            raise S3RequestException(f'S3 Downlaod failed with status code {response.status_code}. {response.text}')

    def download_model_and_unzip(self, model_name: str, path: Path) -> None:
        """
        Download the zipped model to the specified path from the s3 bucket.
        """
        # Models are stored in zip files, add zip back to model_name to get object name
        object_url: str = f"{self._bucket_endpoint}/{model_name}.zip"

        response = requests.get(object_url)

        if response.status_code == 200:
            # Save file
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            raise S3RequestException(f"Could not download model named {model_name} from {self._bucket_endpoint}. Failed with status code {response.status_code}")

        self.unzip_zipped_model(Path(path) / f"{model_name}.zip")

    def unzip_zipped_model(self, path_to_zipped: Path):
        """
        Extract a zipped file to the same directory it is in, and delete the original zip file
        """
        with ZipFile(path_to_zipped, 'r') as zipped:
            zipped.extractall(path_to_zipped.parent)
        # delete original zip file
        path_to_zipped.unlink()