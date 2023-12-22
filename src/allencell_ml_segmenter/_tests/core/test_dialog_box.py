import pytest
from unittest.mock import MagicMock
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QWidget,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QPushButton,
)
from allencell_ml_segmenter.core.dialog_box import DialogBox
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt


@pytest.fixture
def dialog(qtbot: QtBot):
    dialog = DialogBox("Test Message")
    yield dialog


def test_init(dialog):
    assert dialog.getMessage() == "Test Message"


def test_set_message(dialog):
    dialog.setMessage("New Message")
    assert dialog.getMessage() == "New Message"


def test_yes_selected(dialog):
    dialog.yes_selected()
    assert dialog.selection is True


def test_no_selected(dialog):
    dialog.no_selected()
    assert dialog.selection is False


def test_button_connections(dialog, qtbot: QtBot):
    # Manually triggering the click event for testing
    qtbot.mouseClick(
        dialog.yes_btn,
        Qt.LeftButton,
    )
    assert dialog.selection is True

    qtbot.mouseClick(
        dialog.no_btn,
        Qt.LeftButton,
    )
    assert dialog.selection is False
