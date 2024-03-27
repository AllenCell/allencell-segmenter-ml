import csv
from pathlib import Path
from aicsimageio import AICSImage


class ImageUtils:

    @staticmethod
    def extract_num_channels_from_image(path: Path):
        img: AICSImage = AICSImage(str(path))
        return img.dims.C

    @staticmethod
    def extract_num_channels_from_csv(path: Path):
        with open(path) as file:
            reader: csv.reader = csv.DictReader(file)
            # first column contrains files of interest (zeroth column is index)
            line_data_path: Path = Path(next(reader)["raw"])
            return ImageUtils.extract_num_channels_from_image(line_data_path)