from allencell_ml_segmenter.core.publisher import Publisher


class ThresholdingModel(Publisher):
    """
    Stores state relevant to thresholding processes.
    """

    def __init__(self) -> None:
        super().__init__()

        # default value of .5, range from 0.0 to 1.0 inclusive
        self._thresholding_value_selected: float = 0.5


    def set_thresholding_value(self, value: float) -> None:
        """
        Set the thresholding value.
        """
        # Ensure value is within range, the UI should enforce values between 0.0 and 1.0
        if value < 0.0 or value > 1.0:
            raise ValueError("Thresholding value selected must be between 0.0 and 1.0.")
        self._thresholding_value_selected = value


    def get_thresholding_value(self) -> float:
        """
        Get the thresholding value.
        """
        return self._thresholding_value_selected

