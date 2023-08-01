import pytest
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


@pytest.fixture
def label_with_hint(qtbot: QtBot) -> LabelWithHint:
    return LabelWithHint()


def test_set_label_text_updates_label_text(
    label_with_hint: LabelWithHint,
) -> None:
    # Act
    label_with_hint.set_label_text("Hello")

    # Assert
    assert label_with_hint._label.text() == "Hello"


def test_set_hint_updates_question_mark_tooltip(
    label_with_hint: LabelWithHint,
) -> None:
    # Act
    label_with_hint.set_hint("This is a hint")

    # Assert
    assert label_with_hint._question_mark.toolTip() == "This is a hint"


def test_layout_contains_label_and_question_mark(
    label_with_hint: LabelWithHint,
) -> None:
    # Act/Assert
    assert label_with_hint.layout().indexOf(label_with_hint._label) != -1
    assert (
        label_with_hint.layout().indexOf(label_with_hint._question_mark) != -1
    )
