import pytest
from typing import Tuple
from allencell_ml_segmenter.curation.main_view import CurationMainView
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import (
    FakeExperimentsModel,
)
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import CurationService
from allencell_ml_segmenter.curation.curation_image_loader import (
    FakeCurationImageLoader,
)

from unittest.mock import Mock
from pytestqt.qtbot import QtBot
from pathlib import Path


# Danny: possibly replace the fixture with the function below to address the TODO?
@pytest.fixture
def curation_main_view(qtbot: QtBot) -> CurationMainView:
    # TODO #161: refactor, dont mutate fixture in tests below
    curation_model: CurationModel = CurationModel()
    experiments_model: FakeExperimentsModel = FakeExperimentsModel()
    curation_service: Mock = Mock(spec=CurationService)
    return CurationMainView(curation_model, curation_service)


def get_test_instances() -> (
    Tuple[CurationModel, CurationService, CurationMainView]
):
    curation_model = CurationModel()
    curation_model.set_image_loader(
        FakeCurationImageLoader(
            [Path("raw 1"), Path("raw 2"), Path("raw 3")],
            [Path("seg1 1"), Path("seg1 2"), Path("seg1 3")],
            [Path("seg2 1"), Path("seg2 2"), Path("seg2 3")],
        )
    )
    curation_service = CurationService(curation_model, FakeViewer())
    return (
        curation_model,
        curation_service,
        CurationMainView(curation_model, curation_service),
    )


def test_curation_setup(curation_main_view: CurationMainView) -> None:
    # Danny: this test currently only tests that main_view calls init_progress_bar
    # when main_view.curation_setup(first_setup=True) is called. See comments below
    # for more info
    # Alternatives: test that certain buttons are enabled/disabled after the call?
    # This will require reaching into the state of main_view, though

    # Arrange
    curation_main_view._curation_service.build_raw_images_list = Mock(
        return_value=[Path("path_raw")]
    )
    curation_main_view._curation_service.build_seg1_images_list = Mock(
        return_value=[Path("path_seg1")]
    )
    curation_main_view.init_progress_bar = Mock()

    # Act
    curation_main_view.curation_setup(first_setup=True)

    # Assert
    curation_main_view.init_progress_bar.assert_called_once()
    # Danny: AFAIK, the method we want here is 'assert_called_once_with'; since 'called_once_with'
    # doesn't exist, this just returns another Mock and nothing is asserted. Regardless, I think this
    # type of thing should be in the tests for CurationService.curation_setup, not MainView.curation_setup
    curation_main_view._curation_service.add_image_to_viewer.called_once_with(
        ["path_raw"], "[raw] path_raw"
    )
    curation_main_view._curation_service.add_image_to_viewer.called_once_with(
        ["path_seg1"], "[raw] path_seg1"
    )


def test_init_progress_bar() -> None:
    # Danny: this is kind of the pattern I'm thinking we follow, where we examine
    # elements of the UI via main_view fields.. this should only break when the UI
    # changes, which makes some amount of sense for UI tests
    # Arrange
    _, _, main_view = get_test_instances()
    # Act
    main_view.init_progress_bar()

    # Assert
    assert main_view.progress_bar.value() == 1


@pytest.mark.skip
def test_next_image(curation_main_view: CurationMainView) -> None:
    # Danny: this one mainly tests that MainView._next_image calls
    # CurationService.next_image
    # Alternatives: not sure on this... maybe we could use QtBot (I think I've seen that somewhere before)
    # to simulate click on next button
    # and then check status of UI components that should update? Again, requires reaching into
    # MainView state in some way
    # Arrange
    curation_main_view._update_curation_record = Mock()
    curation_main_view.raw_images = [None, Path("path_raw")]
    curation_main_view.seg1_images = [None, Path("path_seg1")]
    assert curation_main_view._curation_model.get_curation_index() == 0

    # Act
    curation_main_view._next_image()

    # Assert
    curation_main_view._curation_service.next_image.assert_called_once()


@pytest.mark.skip
def test_increment_progress_bar(curation_main_view: CurationMainView) -> None:
    # Danny: as the TODO states, this test may need to be deleted since it's not testing
    # against the public API
    # Arrange
    curation_main_view.init_progress_bar()
    curation_main_view._curation_model.set_raw_images([Path(), Path(), Path()])
    initial_value: int = curation_main_view.progress_bar.value()
    # Act
    # TODO #161: refactor, should be testing against public api
    curation_main_view._increment_progress_bar()
    # Assert
    assert curation_main_view.progress_bar.value() == initial_value + 1


@pytest.mark.skip
def test_stop_increment_progress_bar_when_curation_finished(
    curation_main_view: CurationMainView,
) -> None:
    # Danny: as the TODO states, this test may need to be deleted since it's not testing
    # against the public API
    # Arrange
    curation_main_view.init_progress_bar()
    curation_main_view._curation_model.set_raw_images([Path(), Path(), Path()])
    curation_main_view._curation_model.set_curation_index(3)
    initial_value: int = curation_main_view.progress_bar.value()

    # Act
    # TODO #161: refactor, should be testing against public api
    curation_main_view._increment_progress_bar()
    # Assert
    assert curation_main_view.progress_bar.value() == initial_value
