from qtpy.QtWidgets import QListWidget, QListWidgetItem
from typing import Union, List
from qtpy.QtCore import Qt, Signal


class CheckBoxListWidget(QListWidget):
    """
    A custom QListWidget that allows the user to check/uncheck items in the list.
    Used to display choices for on-screen images in the PredictionView's FileInputWidget.
    """

    checkedSignal: Signal = Signal(int, Qt.CheckState)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.itemEntered.connect(self._show_tool_tip)
        self.itemChanged.connect(self._send_checked_signal)
        self.setMaximumHeight(100)
        self.setStyleSheet("margin-top: 0px")

    def _send_checked_signal(self, item: QListWidgetItem) -> None:
        """
        Handles the itemChanged signal by emitting a checkedSignal that indicates which item was chosen.
        """
        idx: int = self.row(item)
        self.checkedSignal.emit(idx, item.checkState())

    def add_item(self, item: Union[str, QListWidgetItem]) -> None:
        """
        Adds an item to the list.
        """
        if isinstance(item, str):
            item_add: QListWidgetItem = QListWidgetItem(item)
        elif isinstance(item, QListWidgetItem):
            item_add: QListWidgetItem = item
        else:
            raise TypeError(
                f"Item added to CheckBoxListWidget must be a string or QListWidgetItem, but got {type(item)} instead"
            )

        # set checkable and unchecked by default
        item_add.setFlags(item_add.flags() | Qt.ItemIsUserCheckable)
        item_add.setCheckState(Qt.Unchecked)
        super().addItem(item_add)

    def set_all_state(self, state: Qt.CheckState) -> None:
        for i in range(self.count()):
            item: QListWidgetItem = self.item(i)
            if item.checkState() != state:
                item.setCheckState(state)

    def get_checked_rows(self) -> List[int]:
        return self.__get_flag_rows(Qt.Checked)

    def get_unchecked_rows(self) -> List[int]:
        return self.__get_flag_rows(Qt.Unchecked)

    def __get_flag_rows(self, flag: Qt.CheckState) -> List[int]:
        flag_lst: List[int] = []
        for i in range(self.count()):
            item: QListWidgetItem = self.item(i)
            if item.checkState() == flag:
                flag_lst.append(i)

        return flag_lst

    def remove_checked_rows(self) -> None:
        self.__remove_flag_rows(Qt.Checked)

    def remove_unchecked_rows(self) -> None:
        self.__remove_flag_rows(Qt.Unchecked)

    def __remove_flag_rows(self, flag: Qt.CheckState) -> None:
        flag_lst: List[int] = self.__get_flag_rows(flag)
        for i in reversed(flag_lst):
            self.takeItem(i)

    def _show_tool_tip(self, item: QListWidgetItem) -> None:
        text: str = item.text()
        if self.fontMetrics().boundingRect(text).width() > self.width():
            self.setToolTip(text)
        else:
            self.setToolTip("")
