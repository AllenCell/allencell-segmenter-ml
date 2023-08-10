import pytest
from pytestqt.qtbot import QtBot

from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint


@pytest.fixture
def label_with_hint(qtbot: QtBot) -> LabelWithHint:
    return LabelWithHint()


def test_set_label_text_updates_label_text(
    label_with_hint: LabelWithHint,
) -> None:
    """
    Tests that the label text is updated when set_label_text is called.
    """
    # ARRANGE
    label_txt: str = "Text goes here"

    # ACT
    label_with_hint.set_label_text(label_txt)

    # ASSERT
    assert label_with_hint._label.text() == label_txt


def test_set_hint_updates_question_mark_tooltip(
    label_with_hint: LabelWithHint,
) -> None:
    """
    Tests that the question mark tooltip is updated when set_hint is called.
    """
    # ARRANGE
    tooltip: str = "Helpful hint"

    # ACT
    label_with_hint.set_hint(tooltip)

    # ASSERT
    assert label_with_hint._question_mark.toolTip() == tooltip


def test_layout_contains_label_and_question_mark(
    label_with_hint: LabelWithHint,
) -> None:
    """
    Tests that the widget layout contains the both label and question mark.
    """
    # ASSERT
    assert label_with_hint.layout().indexOf(label_with_hint._label) != -1
    assert (
        label_with_hint.layout().indexOf(label_with_hint._question_mark) != -1
    )
