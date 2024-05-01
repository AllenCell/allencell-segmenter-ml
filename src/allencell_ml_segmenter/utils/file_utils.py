from pathlib import Path
from typing import List, Generator
import os
import platform
import subprocess


class FileUtils:

    @staticmethod
    def get_all_files_in_dir_ignore_hidden(dir_path: Path) -> List[Path]:
        # sort alphabetically- default sorting behavior for glob
        all_files: List[Path] = list(sorted(dir_path.glob("*.*")))
        # Ignore hidden files (such as .DS_Store on mac)
        # There's no way to do this with Path.glob filtering or methods so using list comprehension
        return [file for file in all_files if not file.name.startswith(".")]

    @staticmethod
    def get_img_path_from_folder(folder: Path) -> Path:
        """
        Returns path of an image in the folder.
        :param folder: path to a folder containing images
        """
        # we expect user to have the same number of channels for all images in their folders
        # and that only images are stored in those folders

        path_generator: Generator[Path] = folder.glob("*.*")
        image: Path = next(path_generator)
        # ignore hidden files
        while str(image.name).startswith("."):
            image: Path = next(path_generator)
        return image.resolve()

    @staticmethod
    def open_directory_in_window(dir: Path) -> None:
        # for Windows operating systems
        if platform.system() == "Windows":
            os.startfile(dir)
        # for MacOS operating systems
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", dir])
        # for Linux
        else:
            subprocess.Popen(["xdg-open", dir])
