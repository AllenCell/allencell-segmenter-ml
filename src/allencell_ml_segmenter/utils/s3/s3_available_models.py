from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)
from allencell_ml_segmenter.utils.zip_file import (
    IZipFileManager,
    ZipFileManager,
)
from pathlib import Path
import requests



class AvailableModels:
    def __init__(
        self,
        model_file_name: str,
        bucket_endpoint: str,
        zip_file_manager: IZipFileManager = ZipFileManager.global_instance(),
    ):
        self._name: str = model_file_name
        # the url of this AvailableModel on the specified s3 bucket
        self._object_url: str = f"{bucket_endpoint}/{model_file_name}"
        # ZipFileManager handles all zip-file related operations for the model
        self._zipfile: IZipFileManager = zip_file_manager

    def download_model_and_unzip(self, path: Path) -> None:
        """
        Download this AvilableModel from s3 and unzip it into the specified :param path:
        """
        response: requests.Response = requests.get(self._object_url)

        if response.status_code == 200:
            # Write contents of s3:getObject to the path (writing the zip file)
            self._zipfile.write_zip_file(path / self._name, response.content)
            # Unzip the contents of the zipfile where it is located, and delete the .zip
            self._zipfile.unzip_zipped_file_and_delete_zip(path / self._name)
        else:
            # Something went wrong when downloading the object from s3
            raise S3RequestException(
                f"Could not download model named {self._name} from {self._object_url}. Failed with status code {response.status_code}"
            )

    def get_name(self) -> str:
        return self._name

    def get_object_url(self) -> str:
        return self._object_url
