from typing import List

import pytest
from pathlib import Path
from unittest.mock import Mock
from aicsimageio import AICSImage
from napari.layers import Shapes

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_model import CurationModel
from unittest.mock import patch

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel


@pytest.fixture
def curation_model() -> CurationModel:
    return CurationModel()


def test_set_raw_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")

    # Act
    curation_model.set_raw_directory(directory)

    # Assert
    assert curation_model.get_raw_directory() == directory


def test_get_raw_directory() -> None:
    # Arrange
    test_path: Path = Path("test_raw_path")
    model = CurationModel(raw_path=test_path)

    # Act / Assert
    assert model.get_raw_directory() == test_path


def test_set_seg1_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")
    # Act
    curation_model.set_seg1_directory(directory)

    # Assert
    assert curation_model.get_seg1_directory() == directory


def test_get_seg1_directory() -> None:
    # Arrange
    test_path: Path = Path("test_seg1_path")
    model = CurationModel(seg1_path=test_path)

    # Act / Assert
    assert model.get_seg1_directory() == test_path


def test_set_seg2_directory(curation_model: CurationModel) -> None:
    # Arrange
    directory: Path = Path("fake_path")

    # Act
    curation_model.set_seg2_directory(directory)

    # Assert
    assert curation_model.get_seg2_directory() == directory


def test_get_seg2_directory() -> None:
    # Arrange
    test_path: Path = Path("test_seg2_path")
    model = CurationModel(seg2_path=test_path)

    # Act / Assert
    assert model.get_seg2_directory() == test_path


def test_set_raw_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 0

    # Act
    curation_model.set_raw_channel(channel)

    # Assert
    assert curation_model.get_raw_channel() == channel


def test_set_seg1_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 1

    # Act
    curation_model.set_seg1_channel(channel)

    # Assert
    assert curation_model.get_seg1_channel() == channel


def test_set_seg2_channel(curation_model: CurationModel) -> None:
    # Arrange
    channel: int = 2

    # Act
    curation_model.set_seg2_channel(channel)

    # Assert
    assert curation_model.get_seg2_channel() == channel


def test_set_view(curation_model: CurationModel) -> None:
    # Arrange
    with patch.object(CurationModel, "dispatch") as dispatch_mock:
        # Act
        curation_model.set_view()

        # Assert
        dispatch_mock.assert_called_once_with(
            Event.PROCESS_CURATION_INPUT_STARTED
        )


def test_get_merging_mask_base_layer(curation_model):
    # Assert
    assert curation_model.get_merging_mask_base_layer() is None


def test_set_merging_mask_base_layer(curation_model):
    # arrange
    curation_model.set_merging_mask_base_layer("layer_name")

    # Act/Assert
    assert curation_model.merging_mask_base_layer == "layer_name"


def test_get_curation_image_dims(curation_model):
    # Assert
    assert curation_model.get_curation_image_dims() is None


def test_set_curation_image_dims(curation_model):
    # Arrange
    curation_model.set_curation_image_dims((100, 200, 3))

    # Act/Assert
    assert curation_model.curation_image_dims == (100, 200, 3)


def test_get_raw_images(curation_model):
    # Arrange
    sample_paths: List[Path] = [Path("raw1"), Path("raw2"), Path("raw3")]
    curation_model.raw_images = sample_paths

    # Act/assert
    assert curation_model.get_raw_images() == sample_paths


def test_set_raw_images(curation_model):
    # Act
    curation_model.set_raw_images([Path("image1"), Path("image2")])

    # Assert
    assert curation_model.get_raw_images() == [Path("image1"), Path("image2")]


def test_get_current_raw_image(curation_model):
    # Arrange
    curation_model.set_raw_images([Path("image1"), Path("image2")])

    # Act/Assert
    assert curation_model.get_current_raw_image() == Path("image1")


def test_get_seg1_images(curation_model):
    # Arrange
    sample_seg1: List[Path] = [Path("seg1"), Path("seg1_2"), Path("seg1_3")]
    curation_model.seg1_images = sample_seg1

    # Act/Assert
    assert curation_model.get_seg1_images() == sample_seg1


def test_set_seg1_images(curation_model):
    # Act
    curation_model.set_seg1_images([Path("image1"), Path("image2")])

    # Assert
    assert curation_model.get_seg1_images() == [Path("image1"), Path("image2")]


def test_get_current_seg1_image(curation_model):
    # Act
    curation_model.set_seg1_images([Path("image1"), Path("image2")])

    # Assert
    assert curation_model.get_current_seg1_image() == Path("image1")


def test_get_seg2_images(curation_model):
    # Arrange
    sample_seg2: List[Path] = [Path("seg2"), Path("seg2_2"), Path("seg2_3")]
    curation_model.seg2_images = sample_seg2

    # Act/Assert
    assert curation_model.get_seg2_images() == sample_seg2


def test_set_seg2_images(curation_model):
    # Act
    curation_model.set_seg2_images([Path("image1"), Path("image2")])

    # Assert
    assert curation_model.get_seg2_images() == [Path("image1"), Path("image2")]


def test_get_current_seg2_image(curation_model):
    # Arrange
    curation_model.set_seg2_images([Path("image1"), Path("image2")])

    # Act/Assert
    assert curation_model.get_current_seg2_image() == Path("image1")


def test_set_current_merging_mask_path(curation_model):
    # Act
    curation_model.set_current_merging_mask_path(Path("mask_path"))

    # Assert
    assert curation_model.get_current_merging_mask_path() == Path("mask_path")


def test_get_excluding_mask_shape_layers(curation_model):
    # Arrange
    sample_shape_layers = [
        Mock(spec=Shapes),
        Mock(spec=Shapes),
        Mock(spec=Shapes),
        Mock(spec=Shapes),
    ]
    curation_model.set_excluding_mask_shape_layers(sample_shape_layers)

    # Act/Assert
    assert (
        curation_model.get_excluding_mask_shape_layers() == sample_shape_layers
    )


def test_set_excluding_mask_shape_layers(curation_model):
    # Arrange
    layers = [Shapes()]

    # Act
    curation_model.set_excluding_mask_shape_layers(layers)

    # Assert
    assert curation_model.get_excluding_mask_shape_layers() == layers


def test_append_excluding_mask_shape_layer(curation_model):
    # Arrange
    layer = Shapes()

    # Act
    curation_model.append_excluding_mask_shape_layer(layer)

    # Assert
    assert curation_model.get_excluding_mask_shape_layers() == [layer]


def test_get_merging_mask_shape_layers(curation_model):
    # Arrange
    sample_shape_layers = [
        Mock(spec=Shapes),
        Mock(spec=Shapes),
        Mock(spec=Shapes),
        Mock(spec=Shapes),
    ]
    curation_model.set_merging_mask_shape_layers(sample_shape_layers)

    # Act/Assert
    assert (
        curation_model.get_merging_mask_shape_layers() == sample_shape_layers
    )


def test_set_merging_mask_shape_layers(curation_model):
    # Arrange
    layers = [Shapes()]

    # Act
    curation_model.set_merging_mask_shape_layers(layers)

    # Assert
    assert curation_model.get_merging_mask_shape_layers() == layers


def test_append_merging_mask_shape_layer(curation_model):
    # Arrange
    layer = Shapes()

    # Act
    curation_model.append_merging_mask_shape_layer(layer)

    # Assert
    assert curation_model.get_merging_mask_shape_layers() == [layer]


def test_image_available(curation_model):
    # Arrange
    assert not curation_model.image_available()

    curation_model.set_raw_images([Path("image1"), Path("image2")])

    # Act/Assert
    assert curation_model.image_available()


def test_get_curation_index(curation_model):
    # Arrange
    assert curation_model.get_curation_index() == 0
    curation_model.set_curation_index(4)

    # Act/assert
    assert curation_model.get_curation_index() == 4


def test_set_curation_index(curation_model):
    # Arrange
    curation_model.set_curation_index(2)

    # Act/Assert
    assert curation_model.get_curation_index() == 2


def test_append_curation_record(curation_model):
    # Arrange
    record = CurationRecord(
        raw_file="raw",
        seg1="seg1",
        seg2="seg2",
        excluding_mask="exclude",
        merging_mask="merge",
        base_image_index="seg1",
        to_use=True,
    )

    # Act
    curation_model.append_curation_record(record)

    # Assert
    assert curation_model.curation_record == [record]
