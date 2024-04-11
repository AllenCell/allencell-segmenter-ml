from typing import List

import pytest
from pathlib import Path
from unittest.mock import Mock
from napari.layers import Shapes

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_model import CurationModel
from unittest.mock import patch


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


def test_get_merging_mask_base_layer(curation_model: CurationModel) -> None:
    # Assert
    assert curation_model.get_merging_mask_base_layer() is None


def test_set_merging_mask_base_layer(curation_model: CurationModel) -> None:
    # arrange
    curation_model.set_merging_mask_base_layer("layer_name")

    # Act/Assert
    assert curation_model.get_merging_mask_base_layer() == "layer_name"


def test_get_curation_image_dims(curation_model: CurationModel) -> None:
    # Assert
    assert curation_model.get_curation_image_dims() is None


def test_set_curation_image_dims(curation_model: CurationModel) -> None:
    # Arrange
    curation_model.set_curation_image_dims((100, 200, 3))

    # Act/Assert
    assert curation_model.get_curation_image_dims() == (100, 200, 3)


def test_set_current_merging_mask_path(curation_model: CurationModel) -> None:
    # Act
    curation_model.set_current_merging_mask_path(Path("mask_path"))

    # Assert
    assert curation_model.get_current_merging_mask_path() == Path("mask_path")


def test_get_excluding_mask_shape_layers(
    curation_model: CurationModel,
) -> None:
    # Arrange
    sample_shape_layers: List[Shapes] = [
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


def test_set_excluding_mask_shape_layers(
    curation_model: CurationModel,
) -> None:
    # Arrange
    layers: List[Shapes] = [Shapes()]

    # Act
    curation_model.set_excluding_mask_shape_layers(layers)

    # Assert
    assert curation_model.get_excluding_mask_shape_layers() == layers


def test_append_excluding_mask_shape_layer(
    curation_model: CurationModel,
) -> None:
    # Arrange
    layer: Shapes = Shapes()

    # Act
    curation_model.append_excluding_mask_shape_layer(layer)

    # Assert
    assert curation_model.get_excluding_mask_shape_layers() == [layer]


def test_get_merging_mask_shape_layers(curation_model: CurationModel) -> None:
    # Arrange
    sample_shape_layers: List[Shapes] = [
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


def test_set_merging_mask_shape_layers(curation_model: CurationModel) -> None:
    # Arrange
    layers: List[Shapes] = [Shapes()]

    # Act
    curation_model.set_merging_mask_shape_layers(layers)

    # Assert
    assert curation_model.get_merging_mask_shape_layers() == layers


def test_append_merging_mask_shape_layer(
    curation_model: CurationModel,
) -> None:
    # Arrange
    layer: Shapes = Shapes()

    # Act
    curation_model.append_merging_mask_shape_layer(layer)

    # Assert
    assert curation_model.get_merging_mask_shape_layers() == [layer]


def test_append_curation_record(curation_model: CurationModel) -> None:
    # Arrange
    record: CurationRecord = CurationRecord(
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
    assert curation_model.get_curation_record() == [record]
