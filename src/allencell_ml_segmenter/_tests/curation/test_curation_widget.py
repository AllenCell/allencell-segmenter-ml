from unittest.mock import Mock
from dataclasses import dataclass
from pathlib import Path

import pytest
import napari

from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    CurationView,
)
from allencell_ml_segmenter.curation.input_view import CurationInputView
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.curation.curation_image_loader import (
    FakeCurationImageLoaderFactory,
)
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.curation.curation_widget import CurationWidget
import allencell_ml_segmenter

IMG_DIR_PATH = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "img_folder"
)

IMG_DIR_FILES = [path for path in IMG_DIR_PATH.iterdir()]


@dataclass
class TestEnvironment:
    model: CurationModel
    widget: CurationWidget


@pytest.fixture
def test_env() -> TestEnvironment:
    model: CurationModel = CurationModel(
        FakeExperimentsModel(), FakeCurationImageLoaderFactory()
    )
    return TestEnvironment(model, CurationWidget(FakeViewer(), model))


def test_view_change(qtbot: QtBot, test_env: TestEnvironment) -> None:
    # Arrange
    test_env.model.set_raw_directory_paths(IMG_DIR_FILES)
    test_env.model.set_seg1_directory_paths(IMG_DIR_FILES)
    test_env.model.set_seg2_directory_paths(IMG_DIR_FILES)

    # Act / Assert
    assert test_env.widget.get_view() == CurationView.INPUT_VIEW
    test_env.model.set_current_view(CurationView.MAIN_VIEW)
    assert test_env.widget.get_view() == CurationView.MAIN_VIEW
