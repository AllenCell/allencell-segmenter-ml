import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QListWidgetItem
from qtpy.QtCore import Qt

from allencell_ml_segmenter.widgets.check_box_list_widget import (
    CheckBoxListWidget,
)


@pytest.fixture
def check_box_list_widget(qtbot: QtBot) -> CheckBoxListWidget:
    """
    Returns a CheckBoxListWidget for testing.
    """
    return CheckBoxListWidget()


def test_add_item(check_box_list_widget: CheckBoxListWidget) -> None:
    """
    Tests that add_item() correctly adds a string or QListWidgetItem to the list.
    """
    # ARRANGE/ACT
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item((QListWidgetItem("2")))

    # ASSERT
    assert check_box_list_widget.count() == 2
    assert check_box_list_widget.item(0).text() == "1"

    # ASSERT - see if string converted to QListWidgetItem
    assert isinstance(check_box_list_widget.item(1), QListWidgetItem)
    assert check_box_list_widget.item(1).text() == "2"


def test_add_item_invalid_throws_error(
    check_box_list_widget: CheckBoxListWidget,
) -> None:
    """
    Tests that adding an item of type other than string or QListWidgetItem throws a TypeError.
    """
    with pytest.raises(TypeError):
        # ACT
        check_box_list_widget.add_item(3)


def test_set_all_state_uniform(
    check_box_list_widget: CheckBoxListWidget,
) -> None:
    """
    Tests that set_all_state() correctly sets all checkboxes to the given state.
    Checkboxes arranged to have a mix of checked and unchecked states.
    """
    # ARRANGE
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.item(0).setCheckState(Qt.Checked)
    check_box_list_widget.item(1).setCheckState(Qt.Unchecked)

    # ACT
    check_box_list_widget.set_all_state(Qt.Checked)

    # ASSERT
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Checked

    # ACT
    check_box_list_widget.set_all_state(Qt.Unchecked)

    # ASSERT
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Unchecked


def test_toggle_state_uniform(
    check_box_list_widget: CheckBoxListWidget,
) -> None:
    """
    Tests that set_all_state() correctly sets all checkboxes to the given state.
    Checkboxes arranged to have uniform checkstates.
    """
    # ARRANGE
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")

    # ACT
    check_box_list_widget.set_all_state(Qt.Checked)

    # ASSERT
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Checked

    # ACT
    check_box_list_widget.set_all_state(Qt.Unchecked)

    # ASSERT
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Unchecked


def test_get_checked_rows(check_box_list_widget: CheckBoxListWidget) -> None:
    """
    Tests that get_checked_rows() returns the correct indices of checked rows.
    """
    # ARRANGE
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")

    # ACT
    check_box_list_widget.item(1).setCheckState(Qt.Checked)
    check_box_list_widget.item(3).setCheckState(Qt.Checked)

    # ASSERT
    assert check_box_list_widget.get_checked_rows() == [1, 3]


def test_get_unchecked_rows(check_box_list_widget: CheckBoxListWidget) -> None:
    """
    Tests that get_unchecked_rows() returns the correct indices of unchecked rows.
    """
    # ARRANGE
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")

    # ACT
    check_box_list_widget.item(1).setCheckState(Qt.Checked)
    check_box_list_widget.item(3).setCheckState(Qt.Checked)

    # ASSERT
    assert check_box_list_widget.get_unchecked_rows() == [0, 2]


def test_remove_checked_rows(
    check_box_list_widget: CheckBoxListWidget,
) -> None:
    """
    Tests that remove_checked_rows() correctly removes checked rows.
    """
    # ARRANGE
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")

    check_box_list_widget.item(0).setCheckState(Qt.Checked)
    check_box_list_widget.item(2).setCheckState(Qt.Checked)

    # ACT
    check_box_list_widget.remove_checked_rows()

    # ASSERT
    assert check_box_list_widget.count() == 2
    assert check_box_list_widget.item(0).text() == "2"
    assert check_box_list_widget.item(1).text() == "4"


def test_remove_unchecked_rows(
    check_box_list_widget: CheckBoxListWidget,
) -> None:
    """
    Tests that remove_unchecked_rows() correctly removes unchecked rows.
    """
    # ARRANGE
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")

    # index 1 and 3 are checked, so 0 and 2 are not
    # initially boxes are unchecked
    check_box_list_widget.item(1).setCheckState(Qt.Checked)
    check_box_list_widget.item(3).setCheckState(Qt.Checked)

    # ACT
    check_box_list_widget.remove_unchecked_rows()

    # ASSERT
    assert check_box_list_widget.count() == 2
    assert check_box_list_widget.item(0).text() == "2"
    assert check_box_list_widget.item(1).text() == "4"
