from typing import List, Optional

from qtpy.QtWidgets import QComboBox
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QSizePolicy,
    QLabel,
    QFrame,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QProgressBar,
    QRadioButton,
)
from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.main.viewer import IViewer
from allencell_ml_segmenter.curation.curation_model import CurationModel, CurationImageType
from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.curation.stacked_spinner import StackedSpinner
from allencell_ml_segmenter.main.segmenter_layer import ShapesLayer
from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox


from napari.utils.notifications import show_info, show_warning
from copy import deepcopy

MERGING_MASK_LAYER_NAME: str = "Merging Mask"
EXCLUDING_MASK_LAYER_NAME: str = "Excluding Mask"


class CurationMainView(View):
    """
    View for Curation UI
    """

    def __init__(self, curation_model: CurationModel, viewer: IViewer) -> None:
        super().__init__()
        self._curation_model: CurationModel = curation_model
        self._viewer: IViewer = viewer

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

        input_images_label: QLabel = QLabel("Curation Progress")
        frame.layout().addWidget(input_images_label, alignment=Qt.AlignHCenter)

        progress_bar_layout: QHBoxLayout = QHBoxLayout()
        # Button and progress bar on top row
        # self.back_button: QPushButton = QPushButton("◄ Back")
        # self.back_button.setObjectName("big_blue_btn")
        # progress_bar_layout.addWidget(self.back_button, alignment=Qt.AlignLeft)

        # inner progress bar frame and layout
        inner_progress_frame: QFrame = QFrame()
        inner_progress_frame.setLayout(QHBoxLayout())
        inner_progress_frame.setObjectName("frame")
        self.progress_bar: QProgressBar = QProgressBar()
        inner_progress_frame.layout().addWidget(self.progress_bar)
        self.progress_bar_image_count: QLabel = QLabel("0/0")
        inner_progress_frame.layout().addWidget(
            self.progress_bar_image_count, alignment=Qt.AlignRight
        )
        progress_bar_layout.addWidget(inner_progress_frame)
        self.next_button: QPushButton = QPushButton()
        self.next_button.setObjectName("big_blue_btn")
        self.next_button.clicked.connect(self._on_next)
        progress_bar_layout.addWidget(
            self.next_button, alignment=Qt.AlignRight
        )
        self.layout().addLayout(progress_bar_layout)

        self.save_csv_button: QPushButton = QPushButton("Save Curation CSV")
        self.save_csv_button.clicked.connect(self._on_save_curation_csv)
        self.save_csv_button.setObjectName("save_csv_btn")
        self.layout().addWidget(self.save_csv_button)

        self.file_name: QLabel = QLabel()
        self.layout().addWidget(self.file_name, alignment=Qt.AlignHCenter)

        use_image_frame: QFrame = QFrame()
        use_image_frame.setObjectName("frame")
        use_image_frame.setLayout(QHBoxLayout())
        use_image_frame.layout().addWidget(
            QLabel("Use this image for training")
        )
        self.yes_radio: QRadioButton = QRadioButton("Yes")
        self.yes_radio.setChecked(True)
        self.yes_radio.clicked.connect(self._on_yes_radio_clicked)

        use_image_frame.layout().addWidget(self.yes_radio)
        self.no_radio: QRadioButton = QRadioButton("No")
        self.no_radio.clicked.connect(self._on_no_radio_clicked)
        use_image_frame.layout().addWidget(self.no_radio)

        self._use_img_stacked_spinner = StackedSpinner(use_image_frame)
        self.layout().addWidget(
            self._use_img_stacked_spinner, alignment=Qt.AlignHCenter
        )

        optional_text: QLabel = QLabel("OPTIONAL", self)
        optional_text.setObjectName("text_with_vert_padding")
        self.layout().addWidget(optional_text, alignment=Qt.AlignHCenter)

        base_image_layout: QGridLayout = QGridLayout()
        base_combo_label: LabelWithHint = LabelWithHint(
            "Select a base segmentation"
        )
        self.merging_base_combo: QComboBox = QComboBox()
        self.merging_base_combo.addItem("seg1")
        self.merging_base_combo.addItem("seg2")
        self.merging_base_combo.currentIndexChanged.connect(
            lambda idx: self._curation_model.set_base_image(
                self.merging_base_combo.currentText() if idx >= 0 else None
            )
        )
        # trigger the change event above to set initial model state
        self.merging_base_combo.setCurrentIndex(1)
        self.merging_base_combo.setCurrentIndex(0)
        # these empty QLabels are for spacing... unfortunately cannot apply styling
        # to a QLayout directly, just QWidgets
        base_image_layout.addWidget(QLabel(), 0, 0)
        base_image_layout.addWidget(base_combo_label, 1, 0, 1, 2)
        base_image_layout.addWidget(self.merging_base_combo, 1, 2, 1, 2)
        base_image_layout.addWidget(QLabel(), 2, 0)
        self.layout().addLayout(base_image_layout)
        # Label for Merging mask
        merging_mask_label_and_status: QHBoxLayout = QHBoxLayout()
        merging_mask_label: LabelWithHint = LabelWithHint("Merging mask")
        self.merging_mask_status: QLabel = QLabel("Create and draw mask")
        merging_mask_label_and_status.addWidget(merging_mask_label)
        merging_mask_label_and_status.addWidget(self.merging_mask_status)
        self.layout().addLayout(merging_mask_label_and_status)
        merging_mask_subtext: QLabel = QLabel(
            "Without merging mask, base image will be used for training."
        )
        merging_mask_subtext.setObjectName("subtext")
        self.layout().addWidget(merging_mask_subtext)

        # buttons for merging mask
        merging_mask_buttons: QHBoxLayout = QHBoxLayout()
        self.merging_create_button: QPushButton = QPushButton("+ Create")
        self.merging_create_button.clicked.connect(self._create_merging_mask)
        self.merging_create_button.setObjectName("small_blue_btn")

        self.merging_delete_button: QPushButton = QPushButton("Delete")
        self.merging_delete_button.clicked.connect(self.delete_merging_mask)

        self.merging_save_button: QPushButton = QPushButton("Save")
        self.merging_save_button.setObjectName("small_blue_btn")
        self.merging_save_button.clicked.connect(self.save_merging_mask)

        merging_mask_buttons.addWidget(self.merging_create_button)
        merging_mask_buttons.addWidget(self.merging_delete_button)
        merging_mask_buttons.addWidget(self.merging_save_button)

        self.layout().addLayout(merging_mask_buttons)

        # Labels for excluding mask
        excluding_mask_labels: QHBoxLayout = QHBoxLayout()
        excluding_mask_label: LabelWithHint = LabelWithHint("Excluding mask")
        excluding_mask_labels.addWidget(excluding_mask_label)
        self.excluding_mask_status = QLabel("Create and draw mask")
        excluding_mask_labels.addWidget(
            self.excluding_mask_status, alignment=Qt.AlignLeft
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
        self.excluding_create_button.clicked.connect(
            self._create_excluding_mask
        )
        # propogate button disabled for v1
        # TODO: enable this in v2
        # excluding_propagate_button: QPushButton = QPushButton(
        #    "Propagate in 3D"
        # )
        # excluding_propagate_button.setEnabled(False)

        # TODO: connect the delete button to some functionality
        self.excluding_delete_button: QPushButton = QPushButton("Delete")
        self.excluding_delete_button.clicked.connect(
            self.delete_excluding_mask
        )

        self.excluding_save_button: QPushButton = QPushButton("Save")
        self.excluding_save_button.setObjectName("small_blue_btn")
        self.excluding_save_button.clicked.connect(self.save_excluding_mask)

        excluding_mask_buttons.addWidget(self.excluding_create_button)
        excluding_mask_buttons.addWidget(self.excluding_delete_button)
        excluding_mask_buttons.addWidget(self.excluding_save_button)
        self.layout().addLayout(excluding_mask_buttons)

        self._curation_model.image_loading_finished.connect(
            self._on_first_image_loading_finished
        )

        self._curation_model.saved_to_disk.connect(self._on_saved_to_disk)
        self._set_to_initial_state()

    def doWork(self) -> None:
        print("work")

    def getTypeOfWork(self) -> None:
        print("getwork")

    def showResults(self) -> None:
        print("show result")

    def _set_to_initial_state(self):
        self._set_next_button_to_loading()
        self.disable_all_masks()
        self._use_img_stacked_spinner.start()

    def _on_image_loading_finished(self) -> None:
        self._enable_next_button()

    def _on_first_image_loading_finished(self) -> None:
        self._use_img_stacked_spinner.stop()
        self._update_progress_bar()
        self._add_curr_images_to_widget()
        self._enable_next_button()
        self._curation_model.image_loading_finished.disconnect(
            self._on_first_image_loading_finished
        )
        self._curation_model.image_loading_finished.connect(
            self._on_image_loading_finished
        )

    def _enable_next_button(self) -> None:
        self.next_button.setEnabled(True)
        if self._curation_model.has_next_image():
            self.next_button.setText("Next ►")
        else:
            self.next_button.setText("Finish ►")

    def _set_next_button_to_loading(self) -> None:
        self.next_button.setEnabled(False)
        self.next_button.setText("Loading next...")

    def _add_curr_images_to_widget(self) -> None:
        raw_img_data: ImageData = self._curation_model.get_curr_image_data(CurationImageType.RAW)
        self._viewer.add_image(
            raw_img_data.np_data, f"[raw] {raw_img_data.path.name}"
        )
        seg1_img_data: ImageData = self._curation_model.get_curr_image_data(CurationImageType.SEG1)
        self._viewer.add_image(
            seg1_img_data.np_data, f"[seg1] {seg1_img_data.path.name}"
        )
        if self._curation_model.has_seg2_data():
            seg2_img_data: ImageData = (
                self._curation_model.get_curr_image_data(CurationImageType.SEG2)
            )
            self._viewer.add_image(
                seg2_img_data.np_data, f"[seg2] {seg2_img_data.path.name}"
            )

        self.enable_valid_masks()
        self.file_name.setText(raw_img_data.path.name)
        self.merging_mask_status.setText("Create and draw mask")
        self.excluding_mask_status.setText("Create and draw mask")

    def _on_next(self) -> None:
        """
        Advance to next image set.
        """
        self._viewer.clear_layers()

        if self._curation_model.has_next_image():
            self._set_next_button_to_loading()
            self._curation_model.next_image()
            self._add_curr_images_to_widget()
            # these lines will update UI and model state, must go after
            # a call to next image
            self.yes_radio.click()
            self.merging_base_combo.setCurrentIndex(0)
        else:
            self._on_save_curation_csv()
            self.disable_all_masks()
            self.yes_radio.setEnabled(False)
            self.no_radio.setEnabled(False)
            self.file_name.setText("None")
            self.next_button.setEnabled(False)
            self.next_button.setText("No more images")
            InfoDialogBox(
                "You have reached the end of the dataset, and your curation CSV has been saved.\nPlease switch to the Training tab to start training a model."
            ).exec()

        self._update_progress_bar()

    def _on_save_curation_csv(self) -> None:
        self._curation_model.save_curr_curation_record_to_disk()
        self.save_csv_button.setEnabled(False)

    def _on_saved_to_disk(self, save_successful: bool) -> None:
        self.save_csv_button.setEnabled(True)
        if save_successful:
            show_info("Current progress saved to CSV")
        else:
            show_warning("Failed to save current progress to CSV")

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
        mask_exists: bool = self._viewer.contains_layer(
            MERGING_MASK_LAYER_NAME
        )
        self.merging_save_button.setEnabled(mask_exists)
        self.merging_create_button.setEnabled(True)
        self.merging_delete_button.setEnabled(mask_exists)
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
        mask_exists: bool = self._viewer.contains_layer(
            EXCLUDING_MASK_LAYER_NAME
        )
        self.excluding_save_button.setEnabled(mask_exists)
        self.excluding_create_button.setEnabled(True)
        self.excluding_delete_button.setEnabled(mask_exists)

    def _update_progress_bar(self) -> None:
        """
        update progress bar based on state of image loader
        """
        curr_val: int = self._curation_model.get_curr_image_index() + 1
        num_images: int = self._curation_model.get_num_images()
        self.progress_bar.setMaximum(num_images)
        self.progress_bar.setValue(curr_val)
        # set progress bar hint
        self.progress_bar_image_count.setText(f"{curr_val}/{num_images}")

    def _discard_layer_prompt(self, layer: str) -> None:
        discard_layer_prompt = DialogBox(
            f"There is already a '{layer}' layer in the viewer. Would you like to discard this layer?"
        )
        discard_layer_prompt.exec()
        return discard_layer_prompt.selection

    def _replace_saved_mask_prompt(self, merging_or_excluding: str):
        replace_prompt = DialogBox(
            f"There is already a {merging_or_excluding} mask layer saved. Would you like to overwrite?"
        )
        replace_prompt.exec()
        return replace_prompt.selection

    def _create_merging_mask(self) -> None:
        if self._viewer.contains_layer(MERGING_MASK_LAYER_NAME):
            if not self._discard_layer_prompt(MERGING_MASK_LAYER_NAME):
                return
            self._viewer.remove_layer(MERGING_MASK_LAYER_NAME)

        self._viewer.add_shapes(
            MERGING_MASK_LAYER_NAME, "royalblue", "add_polygon"
        )
        self.merging_save_button.setEnabled(True)
        self.merging_delete_button.setEnabled(True)
        self.merging_mask_status.setText("Draw mask")

    def save_merging_mask(self) -> None:
        if self._curation_model.get_base_image() is None:
            show_info("Please select a base image to merge with")
            return

        merging_mask: Optional[ShapesLayer] = self._viewer.get_shapes(
            MERGING_MASK_LAYER_NAME
        )
        if merging_mask is None:
            show_info("Please create a merging mask layer")
            return

        if self._curation_model.get_merging_mask() is not None:
            if not self._replace_saved_mask_prompt("merging"):
                return

        # deepcopy so that if a user adds more shapes to existing layer, they don't show up in model
        # could change this behavior based on UX input
        self._curation_model.set_merging_mask(deepcopy(merging_mask.data))
        self.merging_mask_status.setText("Merging mask saved")

    def delete_merging_mask(self) -> None:
        merging_mask: Optional[ShapesLayer] = self._viewer.get_shapes(
            MERGING_MASK_LAYER_NAME
        )
        if merging_mask is not None:
            self._viewer.remove_layer(MERGING_MASK_LAYER_NAME)
        self._curation_model.set_merging_mask(None)
        self.merging_save_button.setEnabled(False)
        self.merging_delete_button.setEnabled(False)
        self.merging_mask_status.setText("Merging mask deleted")

    def _create_excluding_mask(self) -> None:
        excluding_mask: Optional[ShapesLayer] = self._viewer.get_shapes(
            EXCLUDING_MASK_LAYER_NAME
        )
        if excluding_mask is not None:
            if not self._discard_layer_prompt(EXCLUDING_MASK_LAYER_NAME):
                return
            self._viewer.remove_layer(EXCLUDING_MASK_LAYER_NAME)

        self._viewer.add_shapes(
            EXCLUDING_MASK_LAYER_NAME, "coral", "add_polygon"
        )
        self.excluding_save_button.setEnabled(True)
        self.excluding_delete_button.setEnabled(True)
        self.excluding_mask_status.setText("Draw mask")

    def save_excluding_mask(self) -> None:
        excluding_mask: Optional[ShapesLayer] = self._viewer.get_shapes(
            EXCLUDING_MASK_LAYER_NAME
        )
        if excluding_mask is None:
            show_info("Please create an excluding mask layer")
            return

        if self._curation_model.get_excluding_mask() is not None:
            if not self._replace_saved_mask_prompt("excluding"):
                return

        self._curation_model.set_excluding_mask(deepcopy(excluding_mask.data))
        self.excluding_mask_status.setText("Excluding mask saved")

    def delete_excluding_mask(self) -> None:
        excluding_mask: Optional[ShapesLayer] = self._viewer.get_shapes(
            EXCLUDING_MASK_LAYER_NAME
        )
        if excluding_mask is not None:
            self._viewer.remove_layer(EXCLUDING_MASK_LAYER_NAME)
        self._curation_model.set_excluding_mask(None)
        self.excluding_save_button.setEnabled(False)
        self.excluding_delete_button.setEnabled(False)
        self.excluding_mask_status.setText("Excluding mask deleted")

    def disable_all_masks(self) -> None:
        self.disable_merging_mask_buttons()
        self.disable_excluding_mask_buttons()

    def enable_valid_masks(self) -> None:
        if self._curation_model.has_seg2_data():
            self.enable_merging_mask_buttons()
        else:
            self.disable_merging_mask_buttons()
        self.enable_excluding_mask_buttons()

    def _on_no_radio_clicked(self) -> None:
        self.disable_all_masks()
        self._curation_model.set_use_image(False)

    def _on_yes_radio_clicked(self) -> None:
        self.enable_valid_masks()
        self._curation_model.set_use_image(True)
