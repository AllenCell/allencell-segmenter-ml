from pathlib import Path
from zipfile import ZipFile

from allencell_ml_segmenter.utils.zip_file import IZipFileManager


class ZipFileManager(IZipFileManager):
    """
    ZipFileManager to manage writing zip files / unzipping zip files
    """

    def __init__(self):
        super().__init__()

    def write_zip_file(self, path: Path, contents: bytes) -> None:
        """
        Write zip file contents :param contents: to :param path:
        """
        # Save file
        with open(path, "wb") as f:
            f.write(contents)

    def unzip_zipped_file_and_delete_zip(self, path_to_zipped: Path) -> None:
        """
        Extract a zipped file to the same directory it is in :param path_to_zipped: 's parent,
        and delete the original zip file at :param path_to_zipped:
        """
        with ZipFile(path_to_zipped, "r") as zipped:
            zipped.extractall(path_to_zipped.parent)
        # delete original zip file
        path_to_zipped.unlink()
