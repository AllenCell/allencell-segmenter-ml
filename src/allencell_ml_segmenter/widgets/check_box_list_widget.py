from qtpy.QtWidgets import QListWidget, QListWidgetItem
from typing import Union
from qtpy.QtCore import Qt, Signal


class CheckBoxListWidget(QListWidget):
    checkedSignal: Signal = Signal(int, Qt.CheckState)

    def __init__(self) -> None:
        super().__init__()
        self.setMouseTracking(True)
        self.itemEntered.connect(self._show_tool_tip)
        self.itemChanged.connect(self._send_checked_signal)
        self.setMaximumHeight(100)
        self.setStyleSheet("margin-top: 0px;")

    def _send_checked_signal(self, item: QListWidgetItem) -> None:
        idx: int = self.row(item)
        self.checkedSignal.emit(idx, item.checkState())

    def add_item(self, item: Union[str, QListWidgetItem]) -> None:
        if isinstance(item, str):
            item_add = QListWidgetItem(item)
        elif isinstance(item, QListWidgetItem):
            item_add = item
        else:
            raise TypeError(
                "Item added to CheckBoxListWidget must be a string or QListWidgetItem, but"
                f"got {type(item)} instead"
            )
        # set checkable and unchecked by default
        item_add.setFlags(item_add.flags() | Qt.ItemIsUserCheckable)
        item_add.setCheckState(Qt.Unchecked)
        super().addItem(item_add)

    def setAllState(self, state) -> None:
        for i in range(self.count()):
            item = self.item(i)
            if item.checkState() != state:
                item.setCheckState(state)

    def getCheckedRows(self) -> None:
        return self.__getFlagRows(Qt.Checked)

    def getUncheckedRows(self) -> None:
        return self.__getFlagRows(Qt.Unchecked)

    def __getFlagRows(self, flag: Qt.CheckState) -> None:
        flag_lst = []
        for i in range(self.count()):
            item = self.item(i)
            if item.checkState() == flag:
                flag_lst.append(i)

        return flag_lst

    def removeCheckedRows(self) -> None:
        self.__removeFlagRows(Qt.Checked)

    def removeUncheckedRows(self) -> None:
        self.__removeFlagRows(Qt.Unchecked)

    def __removeFlagRows(self, flag) -> None:
        flag_lst = self.__getFlagRows(flag)
        flag_lst = reversed(flag_lst)
        for i in flag_lst:
            self.takeItem(i)

    def _show_tool_tip(self, item: QListWidgetItem) -> None:
        text = item.text()
        if self.fontMetrics().boundingRect(text).width() > self.width():
            self.setToolTip(text)
        else:
            self.setToolTip("")
