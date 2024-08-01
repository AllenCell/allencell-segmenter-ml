from typing import Optional

from qtpy.QtWidgets import QComboBox
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QLabel,
    QFrame,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QProgressBar,
    QRadioButton,
    QDialog,
)
from allencell_ml_segmenter.core.dialog_box import DialogBox
from allencell_ml_segmenter._style import Style
from allencell_ml_segmenter.main.viewer import IViewer
from allencell_ml_segmenter.curation.curation_model import (
    CurationModel,
    ImageType,
)
from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.curation.stacked_spinner import StackedSpinner
from allencell_ml_segmenter.main.segmenter_layer import ShapesLayer
from allencell_ml_segmenter.core.info_dialog_box import InfoDialogBox
from allencell_ml_segmenter.main.main_model import MIN_DATASET_SIZE


from napari.utils.notifications import show_info, show_warning
from copy import deepcopy
import numpy as np

MERGING_MASK_LAYER_NAME: str = "Merging Mask"
EXCLUDING_MASK_LAYER_NAME: str = "Excluding Mask"


class CurationMainView(QWidget):
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

        self.save_csv_button: QPushButton = QPushButton(
            "Save Curation Progress"
        )
        self.save_csv_button.clicked.connect(self._on_save_curation_csv)
        self.save_csv_button.setObjectName("save_csv_btn")
        self.layout().addWidget(self.save_csv_button)

        self.file_name: QLabel = QLabel()
        self.layout().addWidget(self.file_name, alignment=Qt.AlignHCenter)

        use_image_frame: QFrame = QFrame()
        use_image_frame.setObjectName("frame")
        use_image_frame.setLayout(QGridLayout())
        use_image_frame.layout().addWidget(
            QLabel("Use this image for training"), 0, 0, 1, 1
        )
        self.yes_radio: QRadioButton = QRadioButton("Yes")
        self.yes_radio.setChecked(True)
        self.yes_radio.clicked.connect(self._on_yes_radio_clicked)

        use_image_frame.layout().addWidget(self.yes_radio, 0, 1, 1, 1)
        self.no_radio: QRadioButton = QRadioButton("No")
        self.no_radio.clicked.connect(self._on_no_radio_clicked)
        use_image_frame.layout().addWidget(self.no_radio, 0, 2, 1, 1)

        self.use_img_help_text = QLabel(
            f"Must select >= {MIN_DATASET_SIZE} images to use"
        )
        self.use_img_help_text.setObjectName("subtext")
        use_image_frame.layout().addWidget(self.use_img_help_text, 2, 0, 1, 1)

        self.use_img_stacked_spinner = StackedSpinner(use_image_frame)
        self.layout().addWidget(
            self.use_img_stacked_spinner, alignment=Qt.AlignHCenter
        )

        optional_text: QLabel = QLabel("OPTIONAL", self)
        optional_text.setObjectName("text_with_vert_padding")
        self.layout().addWidget(optional_text, alignment=Qt.AlignHCenter)

        base_image_layout: QGridLayout = QGridLayout()
        base_combo_label: LabelWithHint = LabelWithHint(
            "Select a base segmentation"
        )
        base_combo_label.set_hint(
            "Areas of the base segmentation that are covered by the merging mask will be overwritten by the other segmentation"
        )
        self.merging_base_combo: QComboBox = QComboBox()
        self.merging_base_combo.addItem("seg1")
        self.merging_base_combo.addItem("seg2")
        self.merging_base_combo.currentIndexChanged.connect(
            lambda idx: self._curation_model.set_base_image(
                self.merging_base_combo.currentText() if idx >= 0 else None
            )
        )

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
        merging_mask_label.set_hint(
            "Indicates areas of the base segmentation that will be overwritten by the other segmentation"
        )
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
        excluding_mask_labels.setContentsMargins(0, 12, 0, 0)
        excluding_mask_label: LabelWithHint = LabelWithHint("Excluding mask")
        excluding_mask_label.set_hint(
            "Indicates areas of the image set (raw, seg(s)) that will be excluded from training"
        )
        excluding_mask_labels.addWidget(excluding_mask_label)
        self.excluding_mask_status = QLabel("Create and draw mask")
        excluding_mask_labels.addWidget(
            self.excluding_mask_status, alignment=Qt.AlignLeft
        )
        self.layout().addLayout(excluding_mask_labels)

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

    def _set_to_initial_state(self):
        self.save_csv_button.setEnabled(False)
        self._set_next_button_to_loading()
        self.disable_all_masks()
        self.disable_radio_buttons()
        self.use_img_stacked_spinner.start()

    def _on_image_loading_finished(self) -> None:
        self._enable_next_button()

    def _on_first_image_loading_finished(self) -> None:
        self.use_img_stacked_spinner.stop()
        self.update_save_csv_button_enabled_state()
        self._update_progress_bar()
        self.add_curr_images_to_widget()
        self._enable_next_button()
        self.update_radio_buttons_enabled_state()
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

    def add_curr_images_to_widget(self) -> None:
        raw_img_data: ImageData = self._curation_model.get_curr_image_data(
            ImageType.RAW
        )
        self._viewer.add_image(
            raw_img_data.np_data, f"[raw] {raw_img_data.path.name}"
        )
        seg1_img_data: ImageData = self._curation_model.get_curr_image_data(
            ImageType.SEG1
        )
        self._viewer.add_labels(
            seg1_img_data.np_data, f"[seg1] {seg1_img_data.path.name}"
        )
        if self._curation_model.has_seg2_data():
            seg2_img_data: ImageData = (
                self._curation_model.get_curr_image_data(ImageType.SEG2)
            )
            self._viewer.add_labels(
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
        if self.yes_radio.isChecked():
            # prompt and conditionally save unsaved masks
            self._check_unsaved_excluding_mask()
            self._check_unsaved_merging_mask()

        self._viewer.clear_layers()

        if self._curation_model.has_next_image():
            self._set_next_button_to_loading()
            self._curation_model.next_image()
            self.add_curr_images_to_widget()
            # these lines will update UI and model state, must go after
            # a call to next image
            self.yes_radio.click()
            self.update_radio_buttons_enabled_state()
            self.update_save_csv_button_enabled_state()
            self.merging_base_combo.setCurrentIndex(0)
        else:
            self._on_save_curation_csv()
            self.disable_all_masks()
            self.disable_radio_buttons()
            self.file_name.setText("None")
            self.next_button.setEnabled(False)
            self.next_button.setText("No more images")
            self._curation_model.stop_loading_images()
            InfoDialogBox(
                "You have reached the end of the dataset, and your curation CSV has been saved.\nPlease switch to the Training tab to start training a model."
            ).exec()

        self._update_progress_bar()

    def _should_prompt_to_save_mask(
        self,
        mask_layer: Optional[ShapesLayer],
        saved_mask: Optional[np.ndarray],
    ) -> bool:
        if mask_layer is not None:
            return (
                len(mask_layer.data) > 0
                if saved_mask is None
                else not np.array_equal(saved_mask, mask_layer.data)
            )
        return False

    def _check_unsaved_mask(
        self,
        curr_mask_layer: Optional[ShapesLayer],
        saved_mask: Optional[np.ndarray],
        mask_type: str,
    ) -> Optional[np.ndarray]:
        if self._should_prompt_to_save_mask(curr_mask_layer, saved_mask):
            save_changes_prompt = DialogBox(
                f"The current {mask_type} mask layer has unsaved changes. Would you like to save these changes?"
            )
            selection: QDialog.DialogCode = save_changes_prompt.exec()
            if selection == QDialog.DialogCode.Accepted:
                return deepcopy(curr_mask_layer.data)
        return saved_mask

    def _check_unsaved_excluding_mask(self) -> None:
        curr_excl_mask_layer: Optional[ShapesLayer] = self._viewer.get_shapes(
            EXCLUDING_MASK_LAYER_NAME
        )
        saved_excl_mask: Optional[np.ndarray] = (
            self._curation_model.get_excluding_mask()
        )
        self._curation_model.set_excluding_mask(
            self._check_unsaved_mask(
                curr_excl_mask_layer, saved_excl_mask, "excluding"
            )
        )

    def _check_unsaved_merging_mask(self) -> None:
        curr_merg_mask_layer: Optional[ShapesLayer] = self._viewer.get_shapes(
            MERGING_MASK_LAYER_NAME
        )
        saved_merg_mask: Optional[np.ndarray] = (
            self._curation_model.get_merging_mask()
        )
        self._curation_model.set_merging_mask(
            self._check_unsaved_mask(
                curr_merg_mask_layer, saved_merg_mask, "merging"
            )
        )

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

    def disable_radio_buttons(self):
        self.yes_radio.setEnabled(False)
        self.no_radio.setEnabled(False)

    def update_radio_buttons_enabled_state(self):
        if (
            self._curation_model.get_max_num_images_to_use()
            <= MIN_DATASET_SIZE
        ):
            self.yes_radio.setEnabled(False)
            self.no_radio.setEnabled(False)
            self.use_img_help_text.setText("Must use all remaining images")
        else:
            self.yes_radio.setEnabled(True)
            self.no_radio.setEnabled(True)

    def update_save_csv_button_enabled_state(self):
        self.save_csv_button.setEnabled(
            self._curation_model.get_num_images_selected_to_use()
            >= MIN_DATASET_SIZE
        )

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
        return discard_layer_prompt.exec() == QDialog.DialogCode.Accepted

    def _replace_saved_mask_prompt(self, merging_or_excluding: str):
        replace_prompt = DialogBox(
            f"There is already a {merging_or_excluding} mask layer saved. Would you like to overwrite?"
        )
        return replace_prompt.exec() == QDialog.DialogCode.Accepted

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
        self.update_save_csv_button_enabled_state()

    def _on_yes_radio_clicked(self) -> None:
        self.enable_valid_masks()
        self._curation_model.set_use_image(True)
        self.update_save_csv_button_enabled_state()
