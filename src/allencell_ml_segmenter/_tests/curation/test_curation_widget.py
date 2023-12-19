from unittest.mock import Mock

import pytest
import napari

from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.input_view import CurationInputView
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.curation.curation_widget import CurationWidget


@pytest.fixture
def curation_model() -> CurationModel:
    return CurationModel()


@pytest.fixture
def experiments_model() -> FakeExperimentsModel:
    return FakeExperimentsModel()


@pytest.fixture
def viewer() -> Mock:
    return Mock(spec=napari.Viewer)


@pytest.fixture
def curation_widget(
    experiments_model: FakeExperimentsModel, viewer: Mock, qtbot: QtBot
) -> CurationWidget:
    return CurationWidget(
        viewer=viewer,
        main_model=MainModel(),
        experiments_model=experiments_model,
    )


def test_initialize_views(
    curation_model: CurationModel, curation_widget: CurationWidget
):
    # Arrange
    input_view: CurationInputView = CurationInputView(
        curation_model, Mock(spec=CurationService)
    )

    # Act
    curation_widget.initialize_view(input_view)

    # Assert
    assert curation_widget.count() == 3  # 2 views from init + 1 for testing
    assert (
        curation_widget.view_to_index[input_view] == 2
    )  # added view should be at index 2


def test_set_view(
    curation_model: CurationModel, curation_widget: CurationWidget
):
    # Arrange
    input_view: CurationInputView = CurationInputView(
        curation_model, Mock(spec=CurationService)
    )
    curation_widget.initialize_view(input_view)

    # Act
    curation_widget.set_view(input_view)

    # Assert
    assert curation_widget.currentWidget() == input_view
