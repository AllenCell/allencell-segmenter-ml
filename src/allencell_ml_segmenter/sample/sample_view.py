from qtpy.QtWidgets import QVBoxLayout

from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.sample.sample_state_widget import SampleStateWidget
from typing import List

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QSizePolicy,
)

class SampleView(View, Subscriber):
    """
    View that is a subscriber for TrainingWidget, responsible for handling events and updating the models + UI.
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

        # children

        self.btn: QPushButton = QPushButton("Start Training")
        self.btn.clicked.connect(lambda: self._main_model.set_training_running(not self._main_model.get_training_running()))
        self.layout().addWidget(self.btn)

        self.return_btn: QPushButton = QPushButton("Return")
        self.return_btn.clicked.connect(lambda: self._main_model.dispatch(Event.MAIN_SELECTED))
        self.layout().addWidget(self.return_btn)

        self.state_widget = SampleStateWidget(self._main_model)
        layout.addWidget(self.state_widget)

        # events

        self._main_model.subscribe(Event.TRAINING_SELECTED,
                                   self,
                                   lambda e: self._main_model.set_current_view(self))
        
        self._main_model.subscribe(Event.TRAINING, self, 
                             lambda e: self.btn.setText("Stop Training") 
                             if self._main_model.get_training_running() 
                             else self.btn.setText("Start Training"))

    def handle_event(self, event: Event) -> None:
        """
        """