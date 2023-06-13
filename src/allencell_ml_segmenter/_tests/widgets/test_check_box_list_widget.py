import pytest
from unittest.mock import Mock
from qtpy.QtWidgets import QListWidgetItem
from qtpy.QtCore import Qt

from allencell_ml_segmenter.widgets.check_box_list_widget import (
    CheckBoxListWidget,
)


@pytest.fixture
def check_box_list_widget(qtbot):
    return CheckBoxListWidget()


def test_add_item(check_box_list_widget):
    # check_box_list_widget.add_item() accepts strings or QListWidgetItems
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item((QListWidgetItem("2")))

    assert check_box_list_widget.count() == 2
    assert check_box_list_widget.item(0).text() == "1"
    # see if string converted to QListWidgetItem
    assert isinstance(check_box_list_widget.item(1), QListWidgetItem)
    assert check_box_list_widget.item(1).text() == "2"

    with pytest.raises(TypeError):
        check_box_list_widget.add_item(3)


def test_set_all_state_uniform(check_box_list_widget):
    # Testing set_all_state() with all checkboxes checked/unchecked
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")

    check_box_list_widget.setAllState(Qt.Checked)
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Checked

    check_box_list_widget.setAllState(Qt.Unchecked)
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Unchecked

def test_toggle_state_mixed(check_box_list_widget):
    # Testing set_all_state() with mixture of checked/unchecked checkboxes
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")

    check_box_list_widget.setAllState(Qt.Checked)
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Checked

    check_box_list_widget.setAllState(Qt.Unchecked)
    for i in range(check_box_list_widget.count()):
        assert check_box_list_widget.item(i).checkState() == Qt.Unchecked


def test_get_checked_rows(check_box_list_widget):
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")

    # "2" and "4" are checked
    check_box_list_widget.item(1).setCheckState(Qt.Checked)
    check_box_list_widget.item(3).setCheckState(Qt.Checked)

    # Testing Checked Rows
    checked_rows = check_box_list_widget.getCheckedRows()
    # "2" and "4" are at index 1 and 3
    assert checked_rows == [1, 3]


def test_get_unchcked_rows(check_box_list_widget):
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")

    # "2" and "4" are checked
    check_box_list_widget.item(1).setCheckState(Qt.Checked)
    check_box_list_widget.item(3).setCheckState(Qt.Checked)

    unchecked_rows = check_box_list_widget.getUncheckedRows()
    assert unchecked_rows == [0, 2]


def test_remove_checked_rows(check_box_list_widget):
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")
    check_box_list_widget.item(0).setCheckState(Qt.Checked)
    check_box_list_widget.item(2).setCheckState(Qt.Checked)

    check_box_list_widget.removeCheckedRows()

    assert check_box_list_widget.count() == 2
    assert check_box_list_widget.item(0).text() == "2"
    assert check_box_list_widget.item(1).text() == "4"


def test_remove_unchecked_rows(check_box_list_widget):
    check_box_list_widget.add_item("1")
    check_box_list_widget.add_item("2")
    check_box_list_widget.add_item("3")
    check_box_list_widget.add_item("4")
    # index 1 and 3 are checked, so 0 and 2 are not
    # initially boxes are unchecked
    check_box_list_widget.item(1).setCheckState(Qt.Checked)
    check_box_list_widget.item(3).setCheckState(Qt.Checked)

    check_box_list_widget.removeUncheckedRows()

    assert check_box_list_widget.count() == 2
    assert check_box_list_widget.item(0).text() == "2"
    assert check_box_list_widget.item(1).text() == "4"
