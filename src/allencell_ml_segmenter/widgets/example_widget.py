from qtpy.QtWidgets import (
    QWidget, QSizePolicy, QVBoxLayout,
    QHBoxLayout, QCheckBox, QLineEdit,
    QComboBox, QSlider, QPushButton,
    QLabel
)
from qtpy.QtCore import Qt
from typing import List, Callable


class ExampleWidget(QWidget):
    """
    An example widget meant to display various on-screen UI elements.
    """
    def __init__(self):
        super().__init__()

        self.responsive_widgets: List[QWidget] = []

        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        checkbox_layout: QHBoxLayout = QHBoxLayout()

        self.first_option: QCheckBox = QCheckBox("Option 1")
        self.second_option: QCheckBox = QCheckBox("Option 2")
        self.options: List[QCheckBox] = [self.first_option, self.second_option]
        self.responsive_widgets.extend(self.options)

        # Add checkboxes to the same row
        checkbox_layout.addWidget(self.first_option)
        checkbox_layout.addWidget(self.second_option)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)

        # Textbox
        self.textbox: QLineEdit = QLineEdit()
        self.textbox.setPlaceholderText("Type here")

        # Dropdown
        self.dropdown: QComboBox = QComboBox()
        self.dropdown.addItems(["One", "Two", "Three"])

        # Slider
        self.slider: QSlider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.responsive_widgets.append(self.slider)

        # Slider Label
        self.slider_label: QLabel = QLabel("0")
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider_label.setMinimumWidth(80)

        # Slider Layout
        slider_layout: QHBoxLayout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_label)

        # Save Button
        self.save_button: QPushButton = QPushButton("Save")
        self.responsive_widgets.append(self.save_button)

        # Home Button
        self.home_button: QPushButton = QPushButton("Return")
        self.responsive_widgets.append(self.home_button)

        # Add home and save buttons to same row
        button_layout: QHBoxLayout = QHBoxLayout()
        button_layout.addWidget(self.home_button)
        button_layout.addWidget(self.save_button)

        # Add all components to the outermost layout
        self.layout().addWidget(self.textbox)
        self.layout().addWidget(self.dropdown)
        self.layout().addLayout(checkbox_layout)
        self.layout().addLayout(slider_layout)
        self.layout().addLayout(button_layout)

    def connect_slots(self, functions: List[Callable]) -> None:
        """
        changes holds corresponding 'clicked', 'valueChanged', etc.
        Use exec() after building the f-string to run what you want.
        """
        for idx, function in enumerate(functions):
            if idx < 2:
                self.responsive_widgets[idx].stateChanged.connect(function)
            elif idx == 2:
                self.responsive_widgets[idx].valueChanged.connect(function)
            else:
                self.responsive_widgets[idx].clicked.connect(function)
