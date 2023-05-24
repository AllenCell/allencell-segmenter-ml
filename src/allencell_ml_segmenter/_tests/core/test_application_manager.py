import pytest
import napari
from qtpy.QtWidgets import QLayout
from unittest.mock import Mock
from allencell_ml_segmenter.core.router import Router
from allencell_ml_segmenter.core.application_manager import ApplicationManager

@pytest.fixture
def viewer() -> Mock:
    return Mock(spec=napari.Viewer)

@pytest.fixture
def mock_root_layout():
    return Mock(spec=QLayout)

def test_application_manager(viewer, mock_root_layout):
    app_manager = ApplicationManager(viewer, mock_root_layout)
    assert app_manager.viewer == viewer
    assert app_manager.view_manager._base_layout == mock_root_layout
    assert isinstance(app_manager.router, Router)

def test_application_manager_invalid_viewer(mock_root_layout):
    with pytest.raises(ValueError):
        ApplicationManager(None, mock_root_layout)

def test_application_manager_invalid_root_layout(viewer):
    with pytest.raises(ValueError):
        ApplicationManager(viewer, None)