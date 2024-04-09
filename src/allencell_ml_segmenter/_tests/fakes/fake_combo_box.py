from typing import List, Optional


class FakeComboBox:
    def __init__(self):
        self.clear_call_count = 0
        self.items_added: List[int] = []
        self.current_index: Optional[int] = None
        self.enabled = False

    def clear(self) -> None:
        self.clear_call_count = self.clear_call_count + 1

    def addItems(self, items: List[int]) -> None:
        for i in items:
            self.items_added.append(i)

    def setCurrentIndex(self, index: int) -> None:
        self.current_index = index

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled





