from qtpy.QtWidgets import QVBoxLayout

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.sample.sample_state_widget import SampleStateWidget
from allencell_ml_segmenter.sample.sample_results_list_widget import SampleResultsListWidget
from allencell_ml_segmenter.sample.sample_select_files_widget import SampleSelectFilesWidget
from allencell_ml_segmenter.sample.sample_model import SampleModel

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QSizePolicy,
)

from allencell_ml_segmenter.sample.process.service import SampleProcessService

class SampleView(View, Subscriber):
    """
    Sample that orchestrates widgets, managing complex behavior.
    """

    def __init__(self, main_model: MainModel):
        super().__init__()
        self._main_model = main_model

        # init

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        self.setLayout(QVBoxLayout())

        self._sample_model = SampleModel()
        self._service = SampleProcessService(self._sample_model)

        # children

        self._select_files_widget = SampleSelectFilesWidget(self._sample_model)
        self.layout().addWidget(self._select_files_widget)

        self._btn: QPushButton = QPushButton("Start Training")
        self._btn.clicked.connect(lambda: self._service.run())
        self.layout().addWidget(self._btn)

        self.state_widget = SampleStateWidget(self._sample_model)
        layout.addWidget(self.state_widget)
        
        self._results_list_widget = SampleResultsListWidget(self._sample_model)
        self.layout().addWidget(self._results_list_widget)

        self._return_btn: QPushButton = QPushButton("Return")
        self._return_btn.clicked.connect(lambda: self._main_model.dispatch(Event.VIEW_SELECTION_MAIN))
        self.layout().addWidget(self._return_btn)

        # events

        self._main_model.subscribe(Event.VIEW_SELECTION_TRAINING,
                                   self,
                                   lambda e: self._main_model.set_current_view(self))
        
        self._main_model.subscribe(Event.PROCESS_TRAINING, self, 
                             lambda e: self._btn.setText("Stop Training") 
                             if self._main_model.get_training_running() 
                             else self._btn.setText("Start Training"))

    def handle_event(self, event: Event) -> None:
        """
        """