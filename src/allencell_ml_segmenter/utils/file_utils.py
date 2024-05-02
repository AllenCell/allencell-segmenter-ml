import numpy as np
from pathlib import Path
import csv
from typing import List, Generator
from napari.layers import Shapes
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
import os
import platform
import subprocess


# TODO: decide whether utils should be like this with staticmethods or singletons with
# global instances... latter is easier to mock, could combine this with AICSImageData extractor
# e.g.
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
    def write_mask_data(mask: np.ndarray, path: Path):
        """
        Saves the numpy mask data in mask to path
        :param mask: mask to save
        :param path: path at which to save
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)

    @staticmethod
    def write_curation_record(
        curation_record: List[CurationRecord],
        csv_path: Path,
        mask_dir_path: Path,
    ) -> None:
        """
        Saves the curation record as a csv at csv_path and associated masks under mask_dir_path
        :param curation_record: record to save to csv
        :param csv_path: path to save csv
        :param mask_dir_path: directory in which to save masks (masks will be saved under excluding_masks or
        merging_masks subdirs)
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w") as f:
            # need file header
            writer: csv.writer = csv.writer(f, delimiter=",")
            writer.writerow(
                [
                    "",
                    "raw",
                    "seg1",
                    "seg2",
                    "merge_mask",
                    "exclude_mask",
                    "base_image",
                ]
            )

            get_excl_mask_path = (
                lambda raw_path: mask_dir_path
                / "excluding_masks"
                / f"excluding_mask_{raw_path.stem}.npy"
            )
            get_merg_mask_path = (
                lambda raw_path: mask_dir_path
                / "merging_masks"
                / f"merging_mask_{raw_path.stem}.npy"
            )

            idx = 0
            for record in curation_record:
                if record.to_use:
                    if record.excluding_mask is not None:
                        FileUtils.write_mask_data(
                            record.excluding_mask,
                            get_excl_mask_path(record.raw_file.resolve()),
                        )
                    if record.merging_mask is not None:
                        FileUtils.write_mask_data(
                            record.merging_mask,
                            get_merg_mask_path(record.raw_file.resolve()),
                        )

                    writer.writerow(
                        [
                            str(idx),
                            str(record.raw_file.resolve()),
                            str(record.seg1.resolve()),
                            (
                                str(record.seg2.resolve())
                                if record.seg2 is not None
                                else ""
                            ),
                            (
                                get_merg_mask_path(record.raw_file.resolve())
                                if record.merging_mask is not None
                                else ""
                            ),
                            (
                                get_excl_mask_path(record.raw_file.resolve())
                                if record.excluding_mask is not None
                                else ""
                            ),
                            str(record.base_image_index),
                        ]
                    )
                    idx += 1
                f.flush()

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
