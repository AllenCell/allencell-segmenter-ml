import pytest
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint

# app = QApplication([])


@pytest.fixture
def label_with_hint(qtbot):
    return LabelWithHint()


def test_set_label_text_updates_label_text(label_with_hint):
    # Act
    label_with_hint.set_label_text("Hello")

    # Assert
    assert label_with_hint._label.text() == "Hello"


def test_set_hint_updates_question_mark_tooltip(label_with_hint):
    # Act
    label_with_hint.set_hint("This is a hint")

    # Assert
    assert label_with_hint._question_mark.toolTip() == "This is a hint"


def test_layout_contains_label_and_question_mark(label_with_hint):
    # Act/Assert
    assert label_with_hint.layout().indexOf(label_with_hint._label) != -1
    assert (
        label_with_hint.layout().indexOf(label_with_hint._question_mark) != -1
    )
