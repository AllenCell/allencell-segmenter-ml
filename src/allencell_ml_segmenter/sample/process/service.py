from allencell_ml_segmenter.sample.sample_model import SampleModel


class SampleProcessService:
    """
    Abstract class for interoperability between fake and real implementation.
    """

    def __init__(self, sample_model: SampleModel):
        self._sample_model = sample_model

    async def run(self):
        """
        Run the service.

        Args:
            process_model (dict): Process model.
            input_data (dict): Input data.

        Returns:
            dict: Output data.
        """
        if self._sample_model.get_process_running():
            return
        elif (
            self._sample_model.get_training_input_files() is None
            or len(self._sample_model.get_training_input_files()) == 0
        ):
            self._sample_model.set_error_message("No Training File Selected")
        else:
            self._sample_model.set_process_running(True)
            for file in self._sample_model.get_training_input_files():
                self._sample_model.append_training_output_files([file])
            self._sample_model.set_process_running(False)
