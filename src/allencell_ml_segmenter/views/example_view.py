from qtpy.QtWidgets import QVBoxLayout, QMessageBox
from qtpy.QtCore import Qt
from typing import Dict, Any
import os, yaml

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.models.main_model import MainModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.widgets.example_widget import ExampleWidget


class ExampleView(View, Subscriber):
    """
    View that is a subscriber for ExampleWidget, responsible for handling events and updating the models + UI.
    """
    def __init__(self, main_model: MainModel):
        super().__init__()
        layout: QVBoxLayout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # models
        self._main_model: MainModel = main_model
        self._main_model.subscribe(self)

        # init self.widget and connect slots
        self.widget: ExampleWidget = ExampleWidget()
        self.widget.connect_slots([self.on_first_option_click, self.on_second_option_click,
                                  self.update_slider_label, self.on_save_click,
                                  self.back_to_main])
        layout.addWidget(self.widget)

    def handle_event(self, event: Event) -> None:
        """
        Handles Events from the Example Model
        """
        if event == Event.EXAMPLE_SELECTED:
            self._main_model.set_current_view(self)

    def back_to_main(self) -> None:
        """
        Updates models in order to change page back to main.
        """
        self._main_model.dispatch(Event.MAIN_SELECTED)

    def on_first_option_click(self, value: int) -> None:
        """Slot for first checkbox."""
        if value == Qt.Checked:
            self.widget.second_option.setCheckState(Qt.Unchecked)

    def on_second_option_click(self, value: int) -> None:
        """Slot for second checkbox."""
        if value == Qt.Checked:
            self.widget.first_option.setCheckState(Qt.Unchecked)

    def update_slider_label(self, value: int) -> None:
        """Display slider value."""
        self.widget.slider_label.setText(str(value))

    def on_save_click(self) -> None:
        """Slot for the save button. Grabs pertinent values
        and writes current window status to test.yaml (WIP).
        Creates pop-up window to indicates that data has been
        saved. Resets fields to default values.
        """
        status_dict: Dict[str, Any] = {}

        # Ask user to choose a checkbox if none are selected
        if not any(map(lambda x: x.isChecked(), self.widget.options)):
            QMessageBox.warning(self, "Cannot save!", "You must select an option.")
            return

        # Grab text from text box
        status_dict["text"] = self.widget.textbox.text()

        # Grab option from checkboxes
        for i, checkbox in enumerate(self.widget.options):
            if checkbox.isChecked():
                status_dict["option"] = i + 1
                break

        # Grab choice from self.dropdown
        status_dict["choice"] = self.widget.dropdown.currentText()

        # Grab value from slider
        status_dict["slider"] = int(self.widget.slider_label.text())

        # Indicate to the user that their data has been saved
        dialog: QMessageBox = QMessageBox(self)
        dialog.setText("Your information has been saved.")
        dialog.exec_()

        # Reset fields to initial values
        self.widget.textbox.setText("")
        for option in self.widget.options:
            option.setCheckState(Qt.Unchecked)
        self.widget.dropdown.setCurrentIndex(0)
        self.widget.slider.setTracking(True)
        self.widget.slider.setValue(0)
        self.widget.slider.setSliderPosition(0)
        self.widget.slider.update()
        self.widget.slider.repaint()

        # Write status to test.yaml
        if not os.path.isfile("./test.yaml"): # TODO: change file path, "./" refers to desktop rn
            with open("./test.yaml", "w") as file:
                yaml.dump({0: status_dict}, file)
        else:
            with open("./test.yaml", "r") as file:
                previous_entries = yaml.safe_load(file)
                index: int = len(previous_entries)
            with open("./test.yaml", "a") as file:
                yaml.dump({index: status_dict}, file)
