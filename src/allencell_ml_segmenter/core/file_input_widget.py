from pathlib import Path
from typing import List, Optional

from napari.utils.events import Event as NapariEvent  # type: ignore
from napari.utils.notifications import show_warning  # type: ignore
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QRadioButton,
    QComboBox,
    QGridLayout,
    QLabel,
    QFrame,
    QSizePolicy,
    QListWidgetItem,
)

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.widgets.input_button_widget import (
    InputButton,
    FileInputMode,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint

from allencell_ml_segmenter.core.file_input_model import (
    InputMode,
    FileInputModel,
    WidgetMode,
)

from allencell_ml_segmenter.widgets.check_box_list_widget import (
    CheckBoxListWidget,
)

from allencell_ml_segmenter.main.viewer import IViewer
from allencell_ml_segmenter.prediction.service import ModelFileService
from allencell_ml_segmenter.curation.stacked_spinner import StackedSpinner


class FileInputWidget(QWidget):
    """
    A widget containing file inputs for the input to a model prediction.
    """

    TOP_TEXT: str = "On-screen image(s)"
    BOTTOM_TEXT: str = "Image directory"

    def __init__(
        self,
        model: FileInputModel,
        viewer: IViewer,
        service: Optional[ModelFileService],
        include_channel_selection: bool = True,
        mode: WidgetMode = WidgetMode.PREDICTION,
    ):
        super().__init__()
        self._mode: WidgetMode = mode
        self._include_channel_selection: bool = include_channel_selection
        self._model: FileInputModel = model
        self._viewer: IViewer = viewer
        self._service: Optional[ModelFileService] = service
        layout: QVBoxLayout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )

        # set up napari event listener for layer changes
        # this keeps the layer list in our UI updated as the layers are added/deleted from napari
        self._viewer.subscribe_layers_change_event(
            function=self._update_layer_list
        )

        frame: QFrame = QFrame()
        frame_layout: QVBoxLayout = QVBoxLayout()
        frame.setLayout(frame_layout)

        frame.setObjectName("frame")

        title: LabelWithHint = LabelWithHint("Input image(s)")
        title.set_hint("Image(s) to apply the trained model on")
        if self._mode == WidgetMode.THRESHOLDING:
            title.set_hint("Image(s) for thresholding.")
        title.setObjectName("title")

        layout.addWidget(title)
        layout.addWidget(frame)
        frame_layout.addWidget(QLabel("Select input image(s):"))

        # radiobox for images from napari
        horiz_layout: QHBoxLayout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self._radio_on_screen: QRadioButton = QRadioButton()
        self._radio_on_screen.toggled.connect(self._on_screen_slot)
        self._radio_on_screen.setObjectName("onScreen")

        horiz_layout.addWidget(self._radio_on_screen)

        question_label: LabelWithHint = LabelWithHint(FileInputWidget.TOP_TEXT)
        question_label.set_hint("Image(s) already opened in napari")
        horiz_layout.addWidget(question_label)

        frame_layout.addLayout(horiz_layout)

        # list of available images on napari
        self._image_list: CheckBoxListWidget = CheckBoxListWidget()
        self._image_list.setEnabled(False)
        self._image_list.setObjectName("imageList")
        self._image_list.checkedSignal.connect(self._process_checked_signal)
        frame_layout.addWidget(self._image_list)

        # radiobox for images from directory
        horiz_layout = QHBoxLayout()
        horiz_layout.setSpacing(0)

        self._radio_directory: QRadioButton = QRadioButton()
        self._radio_directory.toggled.connect(self._from_directory_slot)
        self._radio_directory.setObjectName("radioDirectory")

        horiz_layout.addWidget(self._radio_directory)

        image_dir_layout: QVBoxLayout = QVBoxLayout()

        question_label = LabelWithHint(FileInputWidget.BOTTOM_TEXT)
        question_label.set_hint(
            "All images in the selected directory will be used as input."
        )

        question_label.add_right_space(10)
        image_dir_layout.addWidget(question_label)

        horiz_layout.addLayout(image_dir_layout)

        horiz_layout.addStretch(5)

        self._browse_dir_edit: InputButton = InputButton(
            self._model,
            lambda dir: self._model.set_input_image_path(
                Path(dir), extract_channels=include_channel_selection
            ),
            "Select directory...",
            FileInputMode.DIRECTORY,
        )
        self.input_image_spinner: StackedSpinner = StackedSpinner(
            input_button=self._browse_dir_edit
        )
        self._browse_dir_edit.setEnabled(False)
        horiz_layout.addWidget(self.input_image_spinner)
        frame_layout.addLayout(horiz_layout)

        grid_layout: QGridLayout = QGridLayout()
        if include_channel_selection:
            image_input_label: LabelWithHint = LabelWithHint(
                "Input image's channel"
            )
            image_input_label.set_hint(
                "Select which channel of the input image(s) to apply the trained model on"
            )

            self._channel_select_dropdown: QComboBox = QComboBox()

            self._channel_select_dropdown.setCurrentIndex(-1)
            self._channel_select_dropdown.currentIndexChanged.connect(
                self._model.set_image_input_channel_index
            )
            self._channel_select_dropdown.setEnabled(False)
            # Event to trigger combobox populate when we know the number of channels
            self._model.subscribe(
                Event.ACTION_FILEINPUT_MAX_CHANNELS_SET,
                self,
                self._populate_input_channel_combobox,
            )
            # Event to set combobox text to 'loading' when we begin extracting channels
            self._model.subscribe(
                Event.ACTION_FILEINPUT_EXTRACT_CHANNELS,
                self,
                self._set_input_channel_combobox_to_loading,
            )
            grid_layout.addWidget(image_input_label, 0, 0)
            grid_layout.addWidget(self._channel_select_dropdown, 0, 1)

        output_dir_label: LabelWithHint = LabelWithHint("Output directory")
        output_dir_label.set_hint("Directory to save result(s) to")

        self._browse_output_edit: InputButton = InputButton(
            self._model,
            lambda dir: self._model.set_output_directory(dir),
            "Select directory...",
            FileInputMode.DIRECTORY,
        )

        grid_layout.addWidget(output_dir_label, 1, 0)
        grid_layout.addWidget(self._browse_output_edit, 1, 1)

        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 0)

        frame_layout.addLayout(grid_layout)

    def _on_screen_slot(self) -> None:
        """Prohibits usage of non-related input fields if top button is checked."""
        self._reset_channel_combobox()
        self._image_list.setEnabled(True)
        self._browse_dir_edit.setEnabled(False)
        self._update_layer_list()
        self._model.set_input_mode(InputMode.FROM_NAPARI_LAYERS)

    def _from_directory_slot(self) -> None:
        """Prohibits usage of non-related input fields if bottom button is checked."""
        self._reset_channel_combobox()
        self._image_list.setEnabled(False)
        self._browse_dir_edit.setEnabled(True)
        self._model.set_input_mode(InputMode.FROM_PATH)

    def _update_layer_list(self, event: Optional[NapariEvent] = None) -> None:
        existing_selection: List[str] = []
        for i in self._image_list.get_checked_rows():
            q_list_widget_item: Optional[QListWidgetItem] = (
                self._image_list.item(i)
            )
            if q_list_widget_item is not None:
                existing_selection.append(q_list_widget_item.text())

        self._image_list.clear()
        self._reset_channel_combobox()
        for layer in self._viewer.get_layers():
            path_of_layer_image: Optional[Path] = self._viewer.get_source_path(
                layer
            )
            if path_of_layer_image:
                self._image_list.add_item(
                    layer.name, set_checked=layer.name in existing_selection
                )

    def _process_checked_signal(self, row: int, state: Qt.CheckState) -> None:
        if self._model.get_input_mode() == InputMode.FROM_NAPARI_LAYERS:
            selected_indices: List[int] = self._image_list.get_checked_rows()
            selected_paths: List[Path] = []
            for i in selected_indices:
                source_path: Optional[Path] = self._viewer.get_source_path(
                    self._viewer.get_layers_nonthreshold()[i]
                )
                if source_path is not None:
                    selected_paths.append(source_path)

            # this will preserve the invariant: the options in the dropdown will be equal
            # to the number of channels in at least one of the selected images (or empty if no images selected)
            if state == Qt.CheckState.Checked:
                if len(selected_paths) <= 10:
                    # only extract if it's the only one checked and it's just been checked;
                    # otherwise we assume they are checking images with same number of channels
                    self._model.set_selected_paths(
                        selected_paths,
                        extract_channels=len(selected_paths) == 1
                        and self._include_channel_selection,
                    )
                else:
                    show_warning(
                        "Cannot predict on > 10 images from the viewer"
                    )
                    qlist_item: Optional[QListWidgetItem] = (
                        self._image_list.item(row)
                    )
                    if qlist_item is not None:
                        qlist_item.setCheckState(Qt.CheckState.Unchecked)
            elif state == Qt.CheckState.Unchecked:
                # could have unselected the img we got channels from originally, so need to re-extract
                # as long as there are still some images selected
                if (
                    len(selected_indices) == 0
                    and self._include_channel_selection
                ):
                    if self._service:
                        self._service.stop_channel_extraction()  # stop so combobox doesn't reset after thread is finished
                    self._reset_channel_combobox()

                self._model.set_selected_paths(
                    selected_paths,
                    extract_channels=len(selected_paths) > 0
                    and self._include_channel_selection,
                )

    def _reset_channel_combobox(self) -> None:
        if self._include_channel_selection:
            self._channel_select_dropdown.clear()
            self._channel_select_dropdown.setPlaceholderText("")
            self._channel_select_dropdown.setCurrentIndex(-1)
            self._channel_select_dropdown.setEnabled(False)

    def _set_input_channel_combobox_to_loading(
        self, event: Optional[Event] = None
    ) -> None:
        if self._include_channel_selection:
            self.input_image_spinner.start()
            self._channel_select_dropdown.clear()
            self._channel_select_dropdown.setPlaceholderText(
                "loading channels..."
            )
            self._channel_select_dropdown.setCurrentIndex(-1)
            self._channel_select_dropdown.setEnabled(False)

    def _populate_input_channel_combobox(
        self, event: Optional[Event] = None
    ) -> None:
        if self._include_channel_selection:
            self.input_image_spinner.stop()
            channels_in_image: Optional[int] = self._model.get_max_channels()
            max_channels: Optional[int] = self._model.get_max_channels()
            self._reset_channel_combobox()
            if (
                channels_in_image is not None
                and channels_in_image > 0
                and max_channels is not None
            ):
                values_range: List[str] = [str(i) for i in range(max_channels)]
                self._channel_select_dropdown.setPlaceholderText(
                    "select a channel index"
                )

                self._channel_select_dropdown.addItems(values_range)
                self._channel_select_dropdown.setEnabled(True)
            else:
                self._channel_select_dropdown.setPlaceholderText(
                    "no channels to select"
                )
