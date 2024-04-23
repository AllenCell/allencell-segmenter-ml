from typing import List

from qtpy.QtWidgets import QComboBox
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
        self.yes_radio.clicked.connect(self.enable_all_masks)
        use_image_frame.layout().addWidget(self.yes_radio)
        self.no_radio: QRadioButton = QRadioButton("No")
        self.no_radio.clicked.connect(self.disable_all_masks)
        use_image_frame.layout().addWidget(self.no_radio)
        self.layout().addWidget(use_image_frame, alignment=Qt.AlignHCenter)

        # Label for Merging mask
        merging_mask_label_and_status: QHBoxLayout = QHBoxLayout()
        merging_mask_label: LabelWithHint = LabelWithHint("Merging mask")
        self.merging_mask_status: QLabel = QLabel()
        merging_mask_label_and_status.addWidget(merging_mask_label)
        merging_mask_label_and_status.addWidget(self.merging_mask_status)
        self.layout().addLayout(merging_mask_label_and_status)
        merging_mask_subtext: QLabel = QLabel(
            "Without merging mask, Seg 1 will be used for training."
        )
        merging_mask_subtext.setObjectName("subtext")
        self.layout().addWidget(merging_mask_subtext)

        # buttons for merging mask
        merging_mask_buttons: QHBoxLayout = QHBoxLayout()
        self.merging_create_button: QPushButton = QPushButton("+ Create")
        self.merging_create_button.clicked.connect(self._curation_service.create_merging_mask_layer)
        self.merging_create_button.setObjectName("small_blue_btn")
        self.merging_base_combo: QComboBox = QComboBox()
        self.merging_base_combo.addItem("Base Image:")
        self.merging_base_combo.addItem("seg1")
        self.merging_base_combo.addItem("seg2")
        self.merging_delete_button: QPushButton = QPushButton("Delete")
        self.merging_save_button: QPushButton = QPushButton("Save")
        self.merging_save_button.setEnabled(False)
        self.merging_save_button.setObjectName("small_blue_btn")
        self.merging_save_button.clicked.connect(self.save_merging_mask)
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

        excluding_mask_subtext: QLabel = QLabel(
            "If performing merging, merge first before excluding mask."
        )
        excluding_mask_subtext.setObjectName("subtext")
        self.layout().addWidget(excluding_mask_subtext)

        # buttons for excluding mask
        excluding_mask_buttons: QHBoxLayout = QHBoxLayout()
        self.excluding_create_button: QPushButton = QPushButton("+ Create")
        self.excluding_create_button.setObjectName("small_blue_btn")
        self.excluding_create_button.clicked.connect(self._create_excluding_mask)
        # propogate button disabled for v1
        # TODO: enable this in v2
        excluding_propagate_button: QPushButton = QPushButton(
            "Propagate in 3D"
        )
        excluding_propagate_button.setEnabled(False)

        self.excluding_delete_button: QPushButton = QPushButton("Delete")
        self.excluding_save_button: QPushButton = QPushButton("Save")
        self.excluding_save_button.setObjectName("small_blue_btn")
        self.excluding_save_button.clicked.connect(self.save_excluding_mask)
        self.excluding_save_button.setEnabled(False)

        excluding_mask_buttons.addWidget(self.excluding_create_button)
        excluding_mask_buttons.addWidget(excluding_propagate_button)
        excluding_mask_buttons.addWidget(self.excluding_delete_button)
        excluding_mask_buttons.addWidget(self.excluding_save_button)
        self.layout().addLayout(excluding_mask_buttons)

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
        # If there is only one segmentation image set for curation disable merging masks.
        if self._curation_model.get_seg2_directory() is None:
            self.disable_merging_mask_buttons()

        if first_setup:
            _ = show_info("Loading curation images")
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
        self.excluding_save_button.setEnabled(False)
        self.excluding_create_button.setEnabled(True)
        self.excluding_delete_button.setEnabled(True)

    def init_progress_bar(self) -> None:
        """
        Initialize progress bar based on number of images to curate, and set progress bar label
        """
        # set progress bar
        num_images: int = (
            self._curation_model.get_image_loader().get_num_images()
        )
        self.progress_bar.setMaximum(num_images)
        # start at 1 to indicate current image shown.
        self.progress_bar.setValue(1)
        # set progress bar hint
        self.progress_bar_image_count.setText(
            f"{self._curation_model.get_image_loader().get_current_index() + 1}/{num_images}"
        )

    def _update_progress_bar(self) -> None:
        """
        update progress bar based on state of image loader
        """
        curr_val: int = (
            self._curation_model.get_image_loader().get_current_index() + 1
        )
        num_images: int = (
            self._curation_model.get_image_loader().get_num_images()
        )
        self.progress_bar.setValue(curr_val)
        # set progress bar hint
        self.progress_bar_image_count.setText(f"{curr_val}/{num_images}")

    def _next_image(self) -> None:
        """
        Update the curation record with the users selection for the current image
        """
        use_this_image: bool = True
        if self.no_radio.isChecked():
            use_this_image = False

        self._curation_service.next_image(use_this_image)
        self.curation_setup()
        self._update_progress_bar()
        self.reset_excluding_mask_status_label()
        self.reset_merging_mask_status_label()

    def _create_merging_mask(self) -> None:
        self._curation_service.create_merging_mask_layer()
        self.merging_save_button.setEnabled(True)

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

    def _create_excluding_mask(self) -> None:
        self._curation_service.create_excluding_mask_layer()
        self.excluding_save_button.setEnabled(True)

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

    def disable_all_masks(self) -> None:
        self.disable_merging_mask_buttons()
        self.disable_excluding_mask_buttons()

    def enable_all_masks(self) -> None:
        self.enable_merging_mask_buttons()
        self.enable_excluding_mask_buttons()
        # redo curation_setup to reset UI to original state
        self.curation_setup(first_setup=False)
