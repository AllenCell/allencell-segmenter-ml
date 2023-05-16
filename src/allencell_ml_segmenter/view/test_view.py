from allencell_ml_segmenter.core.view import View
from allencell_ml_segmenter.view.test_widget import TestWidget
from ._main_template import MainTemplate
from qtpy.QtWidgets import QVBoxLayout

class TestView(View):
    def __init__(self):
        # hook up controller here when ready
        super().__init__(template_class=MainTemplate)

    def load(self, model):
        self._setup_ui()
        print("loaded")

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(TestWidget())