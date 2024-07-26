import os
import platform
import subprocess
import random
from csv import DictReader
from pathlib import Path
from typing import List, Generator, Tuple, Optional

from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.utils.file_writer import IFileWriter
from allencell_ml_segmenter.main.main_model import MIN_DATASET_SIZE

LOSS_COLUMN: str = "val/loss_epoch"


class FileUtils:
    """
    FileUtils handles file reading/writing tasks. In order to use the instance methods (write methods),
    please initialize a FileUtils instance with an IFileWriter object.
    """

    def __init__(self, file_writer: IFileWriter):
        self._file_writer = file_writer

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
    def count_images_in_csv_folder(folder: Path) -> int:
        """
        Given a :param folder: containing train/test/val csvs, returns the
        number of unique images contained in all of the csvs.
        """
        path_generator: Generator[Path] = folder.glob("*.csv")
        images: set[str] = set()
        for p in path_generator:
            with open(p, newline="") as fr:
                reader: DictReader = DictReader(fr)
                for row in reader:
                    if "raw" in row:
                        images.add(str(Path(row["raw"]).resolve()))
        return len(images)

    @staticmethod
    def get_min_loss_from_csv(csv_path: Path) -> Optional[float]:
        min_loss: Optional[float] = None
        with open(csv_path, newline="") as fr:
            reader: DictReader = DictReader(fr)
            if LOSS_COLUMN not in reader.fieldnames:
                return None

            for row in reader:
                entry: str = row[LOSS_COLUMN]
                if len(entry) > 0:
                    loss: float = float(entry)
                    if min_loss is None or loss < min_loss:
                        min_loss = loss

        return min_loss

    def write_curation_record(
        self,
        curation_records: List[CurationRecord],
        csv_dir_path: Path,
        mask_dir_path: Path,
    ) -> None:
        """
        Saves the curation record as a train and test csv in csv_path_dir and associated masks in mask_dir_path
        :param curation_record: record to save to csv
        :param csv_dir_path: directory to save csv (csvs will be named train.csv and test.csv)
        :param mask_dir_path: directory in which to save masks (masks will be saved under excluding_masks or
        merging_masks subdirs)
        """
        train, test = self._train_test_split(curation_records)
        self._write_curation_csv(
            train, csv_dir_path / "train.csv", mask_dir_path
        )
        self._write_curation_csv(
            test, csv_dir_path / "test.csv", mask_dir_path
        )
        self._write_curation_csv(test, csv_dir_path / "val.csv", mask_dir_path)

    def _train_test_split(
        self,
        curation_records: List[CurationRecord],
    ) -> Tuple[List[CurationRecord], List[CurationRecord]]:
        """
        Returns (train_split, test_split) of the items marked to_use in the provided curation record.
        Attempts to reserve 10% of these for test, but if that value is < 2, it will become 2, and
        if it is > 100, it will become 100. If there are < 4 curation records selected for use, an exception will be thrown.
        :param curation_record: record to split
        """
        curation_records_to_use: List[CurationRecord] = [
            r for r in curation_records if r.to_use
        ]
        if len(curation_records_to_use) < MIN_DATASET_SIZE:
            raise RuntimeError(
                f"At least {MIN_DATASET_SIZE} images must be selected for use"
            )

        test_len: int = max(2, min(100, len(curation_records_to_use) // 10))
        random.shuffle(curation_records_to_use)
        return (
            curation_records_to_use[test_len:],
            curation_records_to_use[:test_len],
        )

    def _write_curation_csv(
        self,
        curation_records: List[CurationRecord],
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
        self._file_writer.csv_open_write_mode(csv_path)
        self._file_writer.csv_write_row(
            csv_path,
            [
                "",
                "raw",
                "seg1",
                "seg2",
                "merge_mask",
                "exclude_mask",
                "base_image",
            ],
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
        for record in curation_records:
            if record.to_use:
                if record.excluding_mask is not None:
                    self._file_writer.np_save(
                        get_excl_mask_path(record.raw_file.resolve()),
                        record.excluding_mask,
                    )
                if record.merging_mask is not None:
                    self._file_writer.np_save(
                        get_merg_mask_path(record.raw_file.resolve()),
                        record.merging_mask,
                    )

                self._file_writer.csv_write_row(
                    csv_path,
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
                        str(record.base_image),
                    ],
                )
                idx += 1
        self._file_writer.csv_close(csv_path)

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
