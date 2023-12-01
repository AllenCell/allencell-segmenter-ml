from pathlib import Path
from typing import List

from PyQt5.QtWidgets import QComboBox
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QLabel,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QRadioButton,
)
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import (
    CurationService,
    SelectionMode,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint

from napari.utils.notifications import show_info


class CurationMainView(View):
    """
    View for Curation UI
    """

    def __init__(
        self,
        curation_model: CurationModel,
        curation_service: CurationService,
    ) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._curation_service: CurationService = curation_service
        self.curation_record: List[CurationRecord] = list()
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 10, 0, 10)
        self.layout().setSpacing(0)
        self.layout().setAlignment(Qt.AlignTop)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.setStyleSheet(Style.get_stylesheet("curation_main.qss"))

        self._title: QLabel = QLabel("CURATION UI MAIN", self)
        self._title.setObjectName("title")
        self.layout().addWidget(
            self._title, alignment=Qt.AlignHCenter | Qt.AlignTop
        )

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())
        self.layout().addWidget(frame)

        input_images_label: QLabel = QLabel("Progress")
        frame.layout().addWidget(input_images_label, alignment=Qt.AlignHCenter)

        progress_bar_layout: QHBoxLayout = QHBoxLayout()
        # Button and progress bar on top row
        self.back_button: QPushButton = QPushButton("◄ Back")
        self.back_button.setObjectName("big_blue_btn")
        progress_bar_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)
        # inner progress bar frame and layout
        inner_progress_frame: QFrame = QFrame()
        inner_progress_frame.setLayout(QHBoxLayout())
        inner_progress_frame.setObjectName("frame")
        progress_bar_label: QLabel = QLabel("Image")
        inner_progress_frame.layout().addWidget(
            progress_bar_label, alignment=Qt.AlignLeft
        )
        self.progress_bar: QProgressBar = QProgressBar()
        inner_progress_frame.layout().addWidget(self.progress_bar)
        self.progress_bar_image_count: QLabel = QLabel("0/0")
        inner_progress_frame.layout().addWidget(
            self.progress_bar_image_count, alignment=Qt.AlignRight
        )
        progress_bar_layout.addWidget(inner_progress_frame)
        self.next_button: QPushButton = QPushButton("Next ►")
        self.next_button.setObjectName("big_blue_btn")
        self.next_button.clicked.connect(self._next_image)
        progress_bar_layout.addWidget(
            self.next_button, alignment=Qt.AlignRight
        )
        self.layout().addLayout(progress_bar_layout)

        self.file_name: QLabel = QLabel("Example_file_1")
        self.layout().addWidget(self.file_name, alignment=Qt.AlignHCenter)

        use_image_frame: QFrame = QFrame()
        use_image_frame.setObjectName("frame")
        use_image_frame.setLayout(QHBoxLayout())
        use_image_frame.layout().addWidget(
            QLabel("Use this image for training")
        )
        self.yes_radio: QRadioButton = QRadioButton("Yes")
        self.yes_radio.setChecked(True)
        use_image_frame.layout().addWidget(self.yes_radio)
        self.no_radio: QRadioButton = QRadioButton("No")
        use_image_frame.layout().addWidget(self.no_radio)
        self.layout().addWidget(use_image_frame, alignment=Qt.AlignHCenter)

        # Label for Merging mask
        merging_mask_label_and_status: QHBoxLayout = QHBoxLayout()
        merging_mask_label: LabelWithHint = LabelWithHint("Merging mask")
        self.merging_mask_status: QLabel = QLabel()
        merging_mask_label_and_status.addWidget(merging_mask_label)
        merging_mask_label_and_status.addWidget(self.merging_mask_status)
        self.layout().addLayout(merging_mask_label_and_status)

        # buttons for merging mask
        merging_mask_buttons: QHBoxLayout = QHBoxLayout()
        self.merging_create_button: QPushButton = QPushButton("+ Create")
        self.merging_create_button.clicked.connect(
            lambda x: self._curation_service.enable_shape_selection_viewer(
                mode=SelectionMode.MERGING
            )
        )
        self.merging_create_button.setObjectName("small_blue_btn")
        self.merging_base_combo: QComboBox = QComboBox()
        self.merging_base_combo.addItem("Base Image:")
        self.merging_base_combo.addItem("Seg 1")
        self.merging_base_combo.addItem("Seg 2")
        self.merging_delete_button: QPushButton = QPushButton("Delete")
        self.merging_save_button: QPushButton = QPushButton("Save")
        self.merging_save_button.setEnabled(False)
        self.merging_save_button.setObjectName("small_blue_btn")
        self.merging_save_button.clicked.connect(
            lambda x: self.merging_selection_finished(save=True)
        )
        merging_mask_buttons.addWidget(self.merging_create_button)
        merging_mask_buttons.addWidget(self.merging_base_combo)
        merging_mask_buttons.addWidget(self.merging_delete_button)
        merging_mask_buttons.addWidget(self.merging_save_button)

        self.layout().addLayout(merging_mask_buttons)

        # Labels for excluding mask
        excluding_mask_labels: QHBoxLayout = QHBoxLayout()
        excluding_mask_label: LabelWithHint = LabelWithHint("Excluding mask")
        excluding_mask_labels.addWidget(excluding_mask_label)
        self.excluding_mask_label = QLabel("File name...")
        excluding_mask_labels.addWidget(
            self.excluding_mask_label, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(excluding_mask_labels)

        # buttons for excluding mask
        excluding_mask_buttons: QHBoxLayout = QHBoxLayout()
        self.excluding_create_button: QPushButton = QPushButton("+ Create")
        self.excluding_create_button.setObjectName("small_blue_btn")
        self.excluding_create_button.clicked.connect(
            lambda x: self._curation_service.enable_shape_selection_viewer(
                mode=SelectionMode.EXCLUDING
            )
        )
        # propogate button disabled for v1
        # TODO: enable this in v2
        excluding_propagate_button: QPushButton = QPushButton(
            "Propagate in 3D"
        )
        excluding_propagate_button.setEnabled(False)

        self.excluding_delete_button: QPushButton = QPushButton("Delete")
        self.excluding_save_button: QPushButton = QPushButton("Save")
        self.excluding_save_button.setObjectName("small_blue_btn")
        self.excluding_save_button.clicked.connect(
            lambda x: self.excluding_selection_finished(save=True)
        )
        self.excluding_save_button.setEnabled(False)

        excluding_mask_buttons.addWidget(self.excluding_create_button)
        excluding_mask_buttons.addWidget(excluding_propagate_button)
        excluding_mask_buttons.addWidget(self.excluding_delete_button)
        excluding_mask_buttons.addWidget(self.excluding_save_button)
        self.layout().addLayout(excluding_mask_buttons)

        self._curation_model.subscribe(
            Event.ACTION_CURATION_DRAW_EXCLUDING,
            self,
            lambda e: self.excluding_selection_inprogress(),
        )

        self._curation_model.subscribe(
            Event.ACTION_CURATION_DRAW_MERGING,
            self,
            lambda e: self.merging_selection_inprogress(),
        )

        self._curation_model.subscribe(
            Event.ACTION_CURATION_SAVED_MERGING_MASK,
            self,
            lambda e: self.update_merging_mask_status_label(),
        )

        self._curation_model.subscribe(
            Event.ACTION_CURATION_SAVE_EXCLUDING_MASK,
            self,
            lambda e: self.update_excluding_mask_status_label(),
        )

    def doWork(self) -> None:
        print("work")

    def getTypeOfWork(self) -> None:
        print("getwork")

    def showResults(self) -> None:
        print("show result")

    def curation_setup(self, first_setup: bool = False) -> None:
        """
        Curation setup. Resets the UI to the default state by enabling corresponding buttons and updating the progress bar.

        first_setup (bool): True if first call to curation_setup, false if used to set up subsequent image sets
        """
        _ = show_info("Loading curation images")

        # If there is only one segmentation image set for curation disable merging masks.
        if self._curation_model.get_seg2_directory() is None:
            self.disable_merging_mask_buttons()
        # If there are two segmentation image sets for curation keep merging masks enabled
        # but disable excluding masks until user has merging mask selection ready
        else:
            self.disable_excluding_mask_buttons()

        if first_setup:
            self._curation_service.curation_setup()
            self.init_progress_bar()

    def disable_merging_mask_buttons(self):
        """
        Disable the buttons for merging mask in the UI
        """
        self.merging_save_button.setEnabled(False)
        self.merging_create_button.setEnabled(False)
        self.merging_delete_button.setEnabled(False)
        self.merging_base_combo.setEnabled(False)

    def enable_merging_mask_buttons(self):
        """
        Enable the buttons for merging mask in the UI
        """
        # save button is off to start with
        self.merging_save_button.setEnabled(False)
        self.merging_create_button.setEnabled(True)
        self.merging_delete_button.setEnabled(True)
        self.merging_base_combo.setEnabled(True)

    def disable_excluding_mask_buttons(self):
        """
        disable the buttons for excluding mask in the UI
        """
        self.excluding_save_button.setEnabled(False)
        self.excluding_create_button.setEnabled(False)
        self.excluding_delete_button.setEnabled(False)

    def enable_excluding_mask_buttons(self):
        """
        Enable the buttons for excluding mask in the UI
        """
        self.excluding_save_button.setEnabled(True)
        self.excluding_create_button.setEnabled(True)
        self.excluding_delete_button.setEnabled(True)

    def init_progress_bar(self) -> None:
        """
        Initialize progress bar based on number of images to curate, and set progress bar label
        """
        # set progress bar
        self.progress_bar.setMaximum(
            len(self._curation_model.get_raw_images())
        )
        self.progress_bar.setValue(0)
        # set progress bar hint
        self.progress_bar_image_count.setText(
            f"{self._curation_model.get_curation_index() + 1}/{len(self._curation_model.get_raw_images())}"
        )

    def _increment_progress_bar(self) -> None:
        """
        increment the progress bar by 1
        """
        # update progress bar
        self.progress_bar.setValue(self.progress_bar.value() + 1)
        # set progress bar hint
        self.progress_bar_image_count.setText(
            f"{self._curation_model.get_curation_index() + 1}/{len(self._curation_model.get_raw_images())}"
        )

    def _next_image(self) -> None:
        """
        Update the curation record with the users selection for the current image
        """
        use_this_image: bool = True
        if self.no_radio.isChecked():
            use_this_image = False

        self._curation_service.next_image(use_this_image)
        self.curation_setup()
        self._increment_progress_bar()
        self.reset_excluding_mask_status_label()
        self.reset_merging_mask_status_label()

    def excluding_selection_inprogress(self) -> None:
        """
        Change the UI to state where excluding selection is in progress
        """
        # flip buttons to inprogress state
        self.flip_button_inprogress_state(
            self.excluding_create_button, self.excluding_selection_finished
        )
        # we now have an excluding mask that is the user can save, so enable save button
        self.excluding_save_button.setEnabled(True)

    def excluding_selection_finished(self, save: bool = False):
        """
        Called when excluding selection is finished

        save (bool): True if user wants to save the current excluding mask, false if they dont want to save it quite yet.
        """
        self._curation_service.finished_shape_selection(
            selection_mode=SelectionMode.EXCLUDING
        )
        # flip buttons to original state
        self.flip_button_normal_state(
            self.excluding_create_button,
            lambda x: self._curation_service.enable_shape_selection_viewer(
                mode=SelectionMode.EXCLUDING
            ),
        )
        if save:
            self.save_excluding_mask()

    def merging_selection_inprogress(self) -> None:
        """
        Change the UI to state where merging selection is in progress
        """
        # flip buttons to in progres sstate
        self.flip_button_inprogress_state(
            self.merging_create_button, self.merging_selection_finished
        )
        # we now have a merging mask that is the user can save, so enable save button
        self.merging_save_button.setEnabled(True)

    def merging_selection_finished(self, save: bool = False) -> None:
        """
        Called when merging selection is finished

        save (bool): True if user wants to save the current merging mask, false if they dont want to save it quite yet.
        """
        self._curation_service.finished_shape_selection(
            selection_mode=SelectionMode.MERGING
        )
        # Flip buttons to normal state
        self.flip_button_normal_state(
            self.merging_create_button,
            lambda x: self._curation_service.enable_shape_selection_viewer(
                mode=SelectionMode.MERGING
            ),
        )

        # user has indicated save
        if save:
            self.save_merging_mask()

    def save_merging_mask(self) -> None:
        """
        Wrapper for curation_service.save_merging_mask() for ui interactivity
        """
        if self.merging_base_combo.currentText() == "Base Image:":
            show_info("Please select a base image to merge with")
        else:
            # image saved
            saved: bool = self._curation_service.save_merging_mask(
                self.merging_base_combo.currentText()
            )
            if saved:
                self.enable_excluding_mask_buttons()

    def save_excluding_mask(self) -> None:
        """
        Wrapper for curation_service.save_excluding_mask() for ui interactivity
        """
        self._curation_service.save_excluding_mask()

    def update_merging_mask_status_label(self) -> None:
        """
        Update the merging mask status label when a mask is saved
        """
        self.merging_mask_status.setText("Merging mask saved to experiment.")

    def reset_merging_mask_status_label(self) -> None:
        """
        Reset the merging mask status label to its default state
        """
        self.merging_mask_status.setText("Please select merging mask.")

    def update_excluding_mask_status_label(self) -> None:
        """
        Update the excluding mask status label when a mask is saved
        """
        self.excluding_mask_label.setText(
            "Excluding mask saved to experiment."
        )

    def reset_excluding_mask_status_label(self) -> None:
        """
        Reset the excluding mask status label to its default state
        """
        self.excluding_mask_label.setText("Please select excluding mask.")

    def flip_button_inprogress_state(
        self, button: QPushButton, on_click: callable
    ) -> None:
        """
        flip a button to its mask drawing in-progress state

        button (QPushButton): button to flip
        on_click (callable): function to assign as onclick handler for this button
        """
        button.setText("Finish")
        button.disconnect()
        button.clicked.connect(on_click)

    def flip_button_normal_state(
        self, button: QPushButton, on_click: callable
    ) -> None:
        """
        flip a button to its default state

        button (QPushButton): button to flip
        on_click (callable): function to assign as onclick handler for this button
        """
        button.setText("+ Create")
        button.disconnect()
        button.clicked.connect(on_click)
