from pathlib import Path

from allencell_ml_segmenter.utils.zip_file import IZipFileManager


class FakeZipFileManager(IZipFileManager):
    _instance = None
    written_zip_files: dict[Path, bytes] = {}
    unzipped_files: list[Path] = []

    def write_zip_file(self, path: Path, contents: bytes) -> None:
        self.written_zip_files[path] = contents

    def unzip_zipped_file_and_delete_zip(self, path_to_zipped: Path) -> None:
        self.unzipped_files.append(path_to_zipped)

    @classmethod
    def global_instance(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
