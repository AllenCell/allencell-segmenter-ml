from qtpy.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from typing import List
from allencell_ml_segmenter.prediction.radio_button_entry_widget import (
    RadioButtonEntry,
)


class RadioButtonList(QWidget):
    def __init__(self, descriptions: List[str]):
        super().__init__()

        # TODO: decide on size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)

        self.entries: List[RadioButtonEntry] = []

        for description in descriptions:
            new_widget: RadioButtonEntry = RadioButtonEntry(description)
            self.entries.append(new_widget)
            self.layout().addWidget(new_widget)
