from pathlib import Path
import requests

from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)
from allencell_ml_segmenter.utils.zip_file import (
    IZipFileManager,
    ZipFileManager,
)


class AvailableModel:
    def __init__(
        self,
        model_file_name: str,
        bucket_endpoint: str,
        path_to_save_model: Path,
        zip_file_manager: IZipFileManager = ZipFileManager(),
    ):
        self._name: str = model_file_name
        # the url of this AvailableModel on the specified s3 bucket
        self._object_url: str = f"{bucket_endpoint}/{model_file_name}"
        # ZipFileManager handles all zip-file related operations for the model
        self._zipfile: IZipFileManager = zip_file_manager
        # dir where models should be downloaded to
        self._path_to_store_model: Path = path_to_save_model

    def download_model_and_unzip(self) -> None:
        """
        Download this AvilableModel from s3 and unzip it into the specified :param path:
        """
        continue_download: bool = True
        # check to see if model already exists.
        if (self._path_to_store_model / Path(self.get_name()).stem).exists():
            overwrite_dialog = DialogBox(
                f"{self.get_name()} is already in your experiments folder. Overwrite?"
            )
            overwrite_dialog.exec()
            continue_download = overwrite_dialog.get_selection()

        if continue_download:
            response: requests.Response = requests.get(self._object_url)

            if response.status_code == 200:
                # Write contents of s3:getObject to the path (writing the zip file)
                self._zipfile.write_zip_file(
                    self._path_to_store_model / self._name, response.content
                )
                # Unzip the contents of the zipfile where it is located, and delete the .zip
                self._zipfile.unzip_zipped_file_and_delete_zip(
                    self._path_to_store_model / self._name
                )
            else:
                # Something went wrong when downloading the object from s3
                raise S3RequestException(
                    f"Could not download model named {self._name} from {self._object_url}. Failed with status code {response.status_code}"
                )

    def get_name(self) -> str:
        return self._name

    def get_object_url(self) -> str:
        return self._object_url
