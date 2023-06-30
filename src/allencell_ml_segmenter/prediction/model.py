from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class PredictionModel(Publisher):
    """
    Stores state relevant to prediction processes.
    """

    def __init__(self):
        super().__init__()

        self._file_path: str = None
        self._preprocessing_method: str = None
        self._postprocessing_method: str = None
        self._postprocessing_simple_threshold: float = None
        self._postprocessing_auto_threshold: str = None

    def get_file_path(self) -> str:
        """
        Gets path to model.
        """
        return self._file_path

    def set_file_path(self, path: str) -> None:
        """
        Sets path to model.
        """
        self._file_path = path
        self.dispatch(Event.ACTION_PREDICTION_MODEL_FILE_SELECTED)

    def get_preprocessing_method(self) -> str:
        """
        Gets preprocessing method associated with currently-selected model.
        """
        return self._preprocessing_method

    def set_preprocessing_method(self, method: str) -> None:
        """
        Sets preprocessing method associated with model after the service parses the file.
        """
        self._preprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_PREPROCESSING_METHOD_SELECTED)

    def get_postprocessing_method(self) -> str:
        """
        Gets postprocessing method selected by user.
        """
        return self._postprocessing_method

    def set_postprocessing_method(self, method: str) -> None:
        """
        Sets postprocessing method selected by user.
        """
        self._postprocessing_method = method
        self.dispatch(Event.ACTION_PREDICTION_POSTPROCESSING_METHOD_SELECTED)

    def get_postprocessing_simple_threshold(self) -> float:
        """
        Gets simple threshold selected by user.
        """
        return self._postprocessing_simple_threshold

    def set_postprocessing_simple_threshold_from_slider(
        self, threshold: float
    ) -> None:
        """
        Sets simple threshold selected by user from the slider.
        """
        self._postprocessing_simple_threshold = threshold
        self.dispatch(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_MOVED
        )

    def set_postprocessing_simple_threshold_from_label(
        self, threshold: float
    ) -> None:
        """
        Sets simple threshold input into the label by the user.
        """
        self._postprocessing_simple_threshold = threshold
        self.dispatch(
            Event.ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD_TYPED
        )

    def get_postprocessing_auto_threshold(self) -> str:
        """
        Gets auto threshold selected by user.
        """
        return self._postprocessing_auto_threshold

    def set_postprocessing_auto_threshold(self, threshold: str) -> None:
        """
        Sets auto threshold selected by user.
        """
        self._postprocessing_auto_threshold = threshold
        self.dispatch(
            Event.ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD_SELECTED
        )
