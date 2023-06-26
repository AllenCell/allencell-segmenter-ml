from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class PredictionModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self):
        super().__init__()

        self.file_path: str = None
        self.preprocessing_method: str = None
        self.postprocessing_method: str = None
        self.postprocessing_threshold: float = None

    def get_file_path(self) -> str:
        """
        Gets path to model.
        """
        return self.file_path

    def set_file_path(self, path: str) -> None:
        """
        Sets path to model.
        """
        self.file_path = path
        self.dispatch(Event.ACTION_PREDICTION_MODEL_FILE_SELECTED)

    def get_preprocessing_method(self) -> str:
        """
        Gets preprocessing method associated with currently-selected model.
        """
        return self.preprocessing_method

    def set_preprocessing_method(self, method: str) -> None:
        """
        Sets preprocessing method associated with model after the service parses the file.
        """
        self.preprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_PREPROCESSING_METHOD_SELECTED)

    def get_postprocessing_method(self) -> str:
        """
        Gets postprocessing method selected by user.
        """
        return self.postprocessing_method

    def set_postprocessing_method(self, method: str) -> None:
        """
        Sets postprocessing method selected by user.
        """
        self.postprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_METHOD_SELECTED)

    def get_postprocessing_threshold(self) -> float:
        """
        Gets postprocessing threshold selected by user.
        """
        return self.postprocessing_threshold

    def set_postprocessing_threshold(self, threshold: float) -> None:
        """
        Sets postprocessing threshold selected by user.
        """
        self.postprocessing_threshold = threshold
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_THRESHOLD_SELECTED)
