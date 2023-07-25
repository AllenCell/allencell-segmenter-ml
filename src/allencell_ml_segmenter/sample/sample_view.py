import asyncio

from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.sample.sample_state_widget import SampleStateWidget
from allencell_ml_segmenter.sample.sample_results_list_widget import (
    SampleResultsListWidget,
)
from allencell_ml_segmenter.sample.sample_select_files_widget import (
    SampleSelectFilesWidget,
)
from allencell_ml_segmenter.sample.sample_model import SampleModel

from qtpy.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QSizePolicy,
)

from allencell_ml_segmenter.sample.process.service import SampleProcessService


class SampleView(View):
    """
    Sample that orchestrates widgets, managing complex behavior.
    """

    def __init__(self):
        super().__init__()

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
        self._btn.clicked.connect(lambda: asyncio.run(self._service.run()))
        self.layout().addWidget(self._btn)

        self.state_widget = SampleStateWidget(self._sample_model)
        layout.addWidget(self.state_widget)

        self._results_list_widget = SampleResultsListWidget(self._sample_model)
        self.layout().addWidget(self._results_list_widget)
