from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.core.event import Event


class SampleModel(Publisher):
    """
    Sample model for this application
    """

    def __init__(self):
        super().__init__()
        self._error_message: str = ""
        self._process_running: bool = False
        self._training_input_files: list = []
        self._training_output_files: list = []

    def get_process_running(self) -> bool:
        """
        getter/property for process running
        """
        return self._process_running

    def get_training_input_files(self) -> list:
        """
        getter/property for training input files
        """
        return self._training_input_files

    def get_training_output_files(self) -> list:
        """
        getter/property for training output files
        """
        return self._training_output_files

    def get_error_message(self) -> str:
        """
        getter/property for error message
        """
        return self._error_message

    # Methods for setting state and dispatching events#
    ###################################################

    def set_error_message(self, error_message: str):
        """
        Set the error message in the UI and dispatch a MainEvent
        """
        self._error_message = error_message
        self.dispatch(Event.PROCESS_TRAINING_SHOW_ERROR)

    def set_process_running(self, process_running: bool):
        """
        Set the process running in the UI and dispatch a MainEvent
        """
        self._process_running = process_running
        self.dispatch(Event.PROCESS_TRAINING_CLEAR_ERROR)
        self.dispatch(Event.PROCESS_TRAINING)

    def append_training_output_files(self, training_output_files: list):
        """
        Append to the training output files in the UI and dispatch a MainEvent
        """
        self._training_output_files.extend(training_output_files)
        self.dispatch(Event.PROCESS_TRAINING_CLEAR_ERROR)
        self.dispatch(Event.PROCESS_TRAINING_PROGRESS)

    def append_training_input_files(self, training_input_files: list):
        """
        Append to the training input files in the UI and dispatch a MainEvent
        """
        self._training_input_files.extend(training_input_files)
        self.dispatch(Event.PROCESS_TRAINING_CLEAR_ERROR)

    def set_training_input_files(self, training_input_files: list):
        """
        Set the training input files in the UI and dispatch a MainEvent
        """
        self._training_input_files = training_input_files
        self.dispatch(Event.PROCESS_TRAINING_CLEAR_ERROR)
