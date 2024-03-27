from pathlib import Path
from typing import List


class FileUtils:

    @staticmethod
    def get_all_files_in_dir_ignore_hidden(dir_path: Path) -> List[Path]:
        all_files: List[Path] = list(dir_path.glob("*.*"))
        # Ignore hidden files (such as .DS_Store on mac)
        # There's no way to do this with Path.glob filtering or methods so using list comprehension
        return [file for file in all_files if not file.name.startswith(".")]
