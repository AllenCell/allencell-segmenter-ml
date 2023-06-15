from allencell_ml_segmenter.sample.sample_model import SampleModel
from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.core.subscriber import Subscriber
from qtpy.QtWidgets import QPushButton

class SampleSelectFilesWidget(View, Subscriber):
    """
    Sets training files in the model.
    """


    def __init__(self, model: SampleModel):
        super().__init__()
        self._model = model
        self.btn = QPushButton("Select Files")
        self.btn.clicked.connect(lambda: self._model.set_training_input_files(["/path/to/file1", "/path/to/file2"]))