import csv
from pathlib import Path
from typing import List

from allencell_ml_segmenter.curation.curation_data_class import CurationRecord


class CSVUtils:
    @staticmethod
    def write_curation_csv(
        curation_record: List[CurationRecord], path: Path
    ) -> None:
        """
        Save the curation record as a csv at the specified path. will create parent directories to path as needed.

        curation_record (List[CurationRecord]): record to save to csv
        path (Path): path to save csv
        """
        parent_path: Path = path.parents[0]
        if not parent_path.is_dir():
            parent_path.mkdir(parents=True)

        with open(path, "w") as f:
            # need file header
            writer: csv.writer = csv.writer(f, delimiter=",")
            writer.writerow(
                [
                    "",
                    "raw",
                    "seg1",
                    "seg2",
                    "excluding_mask",
                    "merging_mask",
                    "merging_col",
                ]
            )
            for idx, record in enumerate(curation_record):
                if record.to_use:
                    writer.writerow(
                        [
                            str(idx),
                            str(record.raw_file),
                            str(record.seg1),
                            str(record.seg2) if record.seg2 else "",
                            str(record.excluding_mask),
                            str(record.merging_mask),
                            str(record.base_image_index),
                        ]
                    )
                f.flush()

        # TODO: WRITE ACTUAL VALIDATION AND TEST SETS