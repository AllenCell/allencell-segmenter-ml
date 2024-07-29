from typing import Optional
import webbrowser
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QFrame,
    QGridLayout,
    QComboBox,
    QRadioButton,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QLabel,
)
from qtpy.QtCore import Qt

from allencell_ml_segmenter.config.i_user_settings import IUserSettings
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.utils.s3.s3_bucket_constants import (
    ENABLE_MODEL_DOWNLOADS,
)
from allencell_ml_segmenter.widgets.label_with_hint_widget import LabelWithHint
from allencell_ml_segmenter.widgets.model_download_dialog import (
    ModelDownloadDialog,
)


class ModelSelectionWidget(QWidget):
    """
    A widget for segmentation model selection.
    """

    TITLE_TEXT: str = "Segmentation model"
    TUTORIAL_TEXT: str = "Tutorial"
    GITHUB_TEXT: str = "GitHub"
    FORUM_TEXT: str = "Forum"
    WEBSITE_TEXT: str = "Website"
    EXPERIMENTS_HOME_TEXT: str = "Experiments Home"
    DOWNLOAD_EXPERIMENTS_TEXT: str = "Download Models"

    def __init__(
        self,
        main_model: MainModel,
        experiments_model: IExperimentsModel,
        user_settings: IUserSettings,
    ):
        super().__init__()

        self._main_model: MainModel = main_model
        self._experiments_model: IExperimentsModel = experiments_model
        self._user_settings: IUserSettings = user_settings

        # widget skeleton
        layout: QVBoxLayout = QVBoxLayout()
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._title: LabelWithHint = LabelWithHint(
            ModelSelectionWidget.TITLE_TEXT,
        )
        self._title.set_hint(
            "A type of ML model that separates the structures of interest from their background in a 2D/3D mircroscopy image"
        )
        self._title.setStyleSheet("padding-top: 12px")

        self._model_name_label: QLabel = QLabel()
        self._model_name_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # Help menu, implemented using a combo box to conform to mockup
        self.help_combo_box: QComboBox = QComboBox()
        self.help_combo_box.setFixedWidth(100)
        self.help_combo_box.setPlaceholderText("Help")
        self.help_combo_box.addItems(
            [
                ModelSelectionWidget.TUTORIAL_TEXT,
                ModelSelectionWidget.GITHUB_TEXT,
                ModelSelectionWidget.FORUM_TEXT,
                ModelSelectionWidget.WEBSITE_TEXT,
                ModelSelectionWidget.EXPERIMENTS_HOME_TEXT,
                ModelSelectionWidget.DOWNLOAD_EXPERIMENTS_TEXT,
            ]
        )
        self.help_combo_box.currentTextChanged.connect(
            self._help_combo_handler
        )

        plugin_title_widget_layout: QHBoxLayout = QHBoxLayout()
        plugin_title_widget_layout.addWidget(
            QLabel("Allen Cell & Structure Segmenter - MACHINE LEARNING")
        )
        plugin_title_widget_layout.addWidget(
            self.help_combo_box, alignment=Qt.AlignmentFlag.AlignRight
        )

        layout.setContentsMargins(20, 20, 20, 20)
        layout.addLayout(plugin_title_widget_layout)

        # layout for model labels
        label_widget_layout: QHBoxLayout = QHBoxLayout()
        layout.addLayout(label_widget_layout)
        label_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        label_widget_layout.setSpacing(0)
        label_widget_layout.addWidget(self._title)
        label_widget_layout.addWidget(
            self._model_name_label, alignment=Qt.AlignLeft
        )
        label_widget_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # TODO: hints for widget titles?

        frame: QFrame = QFrame()
        frame.setLayout(QVBoxLayout())
        # uncomment to make frame visible
        # frame.setObjectName("frame")
        self.layout().addWidget(frame)

        # existing model selection components must be initialized before the new/existing model radios
        self._combo_box_existing_models: QComboBox = QComboBox()
        self._combo_box_existing_models.setCurrentIndex(-1)
        self._combo_box_existing_models.setEnabled(False)
        self._combo_box_existing_models.setMinimumWidth(306)
        self._refresh_experiment_options()
        self._combo_box_existing_models.currentTextChanged.connect(
            self._model_combo_handler
        )

        # model selection components
        top_grid_layout: QGridLayout = QGridLayout()

        self._radio_new_model: QRadioButton = QRadioButton()
        self._radio_new_model.toggled.connect(self._model_radio_handler)
        # initialize the radio button and combos / tabs to match the model state
        top_grid_layout.addWidget(self._radio_new_model, 0, 0)

        self._experiment_name_input: QLineEdit = QLineEdit()
        self._experiment_name_input.setPlaceholderText("Name your model")
        self._experiment_name_input.textChanged.connect(
            self._experiment_name_input_handler
        )

        label_new_model: LabelWithHint = LabelWithHint("Start a new model")
        label_new_model.set_hint(
            "Use your own data to train a model from scratch"
        )
        top_grid_layout.addWidget(label_new_model, 0, 1)
        top_grid_layout.addWidget(self._experiment_name_input, 0, 2)

        self._radio_existing_model: QRadioButton = QRadioButton()
        self._radio_existing_model.toggled.connect(self._model_radio_handler)
        # initialize the radio button and combos / tabs to match the model state
        top_grid_layout.addWidget(self._radio_existing_model, 1, 0)
        top_grid_layout.addWidget(
            LabelWithHint(
                label_text="Select an existing model",
                hint="Resume training from pretrained model weights",
            ),
            1,
            1,
        )
        top_grid_layout.addWidget(self._combo_box_existing_models, 1, 2)

        self._apply_change_stacked_widget = QStackedWidget()

        apply_model_layout = QVBoxLayout()
        apply_model_layout.addLayout(top_grid_layout)

        apply_model_widget = QWidget()
        apply_model_widget.setLayout(apply_model_layout)
        self._apply_change_stacked_widget.addWidget(apply_model_widget)
        self._apply_btn: QPushButton = QPushButton("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._handle_apply_model)
        apply_model_layout.addWidget(self._apply_btn)

        frameLayout = QVBoxLayout()
        frameLayout.addWidget(self._apply_change_stacked_widget)
        frame.layout().addLayout(frameLayout)

        self._experiments_model.subscribe(
            Event.ACTION_REFRESH, self, self._handle_process_event
        )

    def _handle_apply_model(self):
        self._experiments_model.apply_experiment_name(
            self._experiment_name_selection
        )
        self._title.set_value_text(
            "    " + self._experiments_model.get_experiment_name()
        )
        self._apply_change_stacked_widget.setVisible(False)

    def _model_combo_handler(self, experiment_name: str) -> None:
        """
        Triggered when the user selects a model from the _combo_box_existing_models.
        Sets the model path in the model.
        """
        self.select_experiment_name(experiment_name)

    def _experiment_name_input_handler(self, text: str) -> None:
        """
        Triggered when the user types in the _experiment_name_input.
        Sets the model name in the model.
        """
        self.select_experiment_name(text)

    def _help_combo_handler(self, text: str) -> None:
        """
        Triggered when the user selects an option from the help combo box.
        Opens the selected help page.
        """
        if text == ModelSelectionWidget.TUTORIAL_TEXT:
            webbrowser.open(
                "https://www.allencell.org/allencell-segmenter-ml-tutorials.html"
            )
        elif text == ModelSelectionWidget.GITHUB_TEXT:
            webbrowser.open(
                "https://github.com/AllenCell/allencell-ml-segmenter"
            )
        elif text == ModelSelectionWidget.FORUM_TEXT:
            webbrowser.open("https://forum.image.sc/tag/segmenter")
        elif text == ModelSelectionWidget.WEBSITE_TEXT:
            webbrowser.open("https://www.allencell.org/segmenter.html")
        elif text == ModelSelectionWidget.EXPERIMENTS_HOME_TEXT:
            self._user_settings.display_change_user_experiments_home(
                parent=self
            )
            self._refresh_experiment_options()
        elif (
            text == ModelSelectionWidget.DOWNLOAD_EXPERIMENTS_TEXT
            and ENABLE_MODEL_DOWNLOADS
        ):
            dialog = ModelDownloadDialog(self, self._experiments_model)
            dialog.exec()
            # once all models are downloaded, one final refresh to load them into existing models dropdown
            self._refresh_experiment_options()

        # reset the combo box, so that it bahaves more like a menu
        self.help_combo_box.setCurrentIndex(-1)

    def _model_radio_handler(self) -> None:
        self._main_model.set_new_model(self._radio_new_model.isChecked())
        self._combo_box_existing_models.setCurrentIndex(-1)
        self._experiment_name_input.clear()
        self.select_experiment_name(None)
        if self._radio_new_model.isChecked():
            """
            Triggered when the user selects the "start a new model" radio button.
            Enables and disables relevent controls.
            """
            self._combo_box_existing_models.setEnabled(False)
            self._experiment_name_input.setEnabled(True)

        if self._radio_existing_model.isChecked():
            """
            Triggered when the user selects the "existing model" radio button.
            Enables and disables relevent controls.
            """
            self._combo_box_existing_models.setEnabled(
                self._experiments_model.get_experiments() != []
            )
            self._experiment_name_input.setEnabled(False)

    def _handle_process_event(self, _: Event = None) -> None:
        """
        Refreshes the experiments in the _combo_box_existing_models.
        """
        if self._radio_new_model.isChecked():
            self._refresh_experiment_options()

    def select_experiment_name(self, name: Optional[str]) -> None:
        """
        Sets experiment name
        """
        if name == "":
            self._experiment_name_selection = None
        else:
            self._experiment_name_selection = name

        experiment_selected = self._experiment_name_selection is not None
        self._apply_btn.setEnabled(experiment_selected)

    def _refresh_experiment_options(self):
        self._experiments_model.refresh_experiments()
        self._combo_box_existing_models.clear()
        self._combo_box_existing_models.addItems(
            self._experiments_model.get_experiments()
        )
        self._combo_box_existing_models.setCurrentIndex(-1)
        self._combo_box_existing_models.setPlaceholderText(
            "No existing models"
            if self._experiments_model.get_experiments() == []
            else "Select an existing model"
        )
