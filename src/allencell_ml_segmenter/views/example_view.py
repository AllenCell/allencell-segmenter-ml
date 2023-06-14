from qtpy.QtWidgets import QVBoxLayout, QMessageBox
from qtpy.QtCore import Qt

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.models.main_model import MainModel
from allencell_ml_segmenter.models.example_model import ExampleModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.widgets.example_widget import ExampleWidget
from allencell_ml_segmenter.services.example_service import ExampleService


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
        self._main_model.subscribe(Event.EXAMPLE_SELECTED, self)

        self._example_model: ExampleModel = ExampleModel()

        # Service - must subscribe to example model before view, or info gets deleted
        self.example_service: ExampleService = ExampleService(
            self._example_model
        )
        self._example_model.subscribe(Event.SAVE, self.example_service)

        # Subscribe view
        self._example_model.subscribe(Event.SAVE, self)

        # init self.widget and connect slots
        self.widget: ExampleWidget = ExampleWidget()
        self.widget.connect_slots(
            [
                self.textbox_changed,
                self.dropdown_changed,
                self.on_first_option_click,
                self.on_second_option_click,
                self.slider_changed,
                self.on_save_click,
                self.back_to_main,
            ]
        )
        layout.addWidget(self.widget)

    def handle_event(self, event: Event) -> None:
        """
        Handles events from both the main and example models. If a save is
        triggered, a pop-up window is created to indicate that data has been
        saved and fields are reset to default values.
        """
        if event == Event.EXAMPLE_SELECTED:
            self._main_model.set_current_view(self)
        elif event == Event.SAVE:
            # Indicate to the user that their data has been saved
            dialog: QMessageBox = QMessageBox(self)
            dialog.setText("Your information has been saved.")
            dialog.exec_()

            # Reset fields to initial values
            self.reset_field_values()

    def textbox_changed(self, s: str) -> None:
        """Textbox (QLineEdit) slot."""
        self._example_model.text = s

    def dropdown_changed(self, s: str) -> None:
        """Dropdown (QComboBox) slot."""
        self._example_model.choice = s

    def on_first_option_click(self, value: int) -> None:
        """Slot for first checkbox."""
        if value == Qt.Checked:
            self.widget.second_option.setCheckState(Qt.Unchecked)
            self._example_model.option = 1

    def on_second_option_click(self, value: int) -> None:
        """Slot for second checkbox."""
        if value == Qt.Checked:
            self.widget.first_option.setCheckState(Qt.Unchecked)
            self._example_model.option = 2

    def slider_changed(self, value: int) -> None:
        """Slider (QSlider) slot."""
        self._example_model.slider = value

        # Display current value
        self.widget.slider_label.setText(str(value))

    def on_save_click(self) -> None:
        """
        Slot for the save button. Has the example model dispatch a save event,
        as long as an option is chosen among the checkboxes.
        """
        # Ask user to choose a checkbox if none are selected
        if not any(map(lambda x: x.isChecked(), self.widget.options)):
            QMessageBox.warning(
                self, "Cannot save!", "You must select an option."
            )
        else:
            self._example_model.save(True)

    def back_to_main(self) -> None:
        """
        Updates models in order to change page back to main.
        """
        self._main_model.dispatch(Event.MAIN_SELECTED)

    def reset_field_values(self) -> None:
        """
        Do not do this before writing values to test.yaml. The reset process
        will trigger valueChanged, stateChanged, etc. events that mistakenly
        update the example model.
        """
        self.widget.textbox.setText("")
        for option in self.widget.options:
            option.setCheckState(Qt.Unchecked)
        self.widget.dropdown.setCurrentIndex(0)
        self.widget.slider.setTracking(True)
        self.widget.slider.setValue(0)
        self.widget.slider.setSliderPosition(0)
        self.widget.slider.update()
        self.widget.slider.repaint()
