from abc import ABC, abstractmethod
from pathlib import Path


class IZipFileManager(ABC):
    """
    ZipFileManager interface
    """

    @abstractmethod
    def write_zip_file(self, path: Path, contents: bytes) -> None:
        """
        Write zip file contents :param contents: to :param path:
        """
        pass

    @abstractmethod
    def unzip_zipped_file_and_delete_zip(self, path_to_zipped: Path) -> None:
        """
        Extract a zipped file to the same directory it is in (:param path_to_zipped: 's parent),
        and delete the original zip file at :param path_to_zipped:
        """
        pass
