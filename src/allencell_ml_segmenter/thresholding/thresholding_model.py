from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class ThresholdingModel(Publisher):
    """
    Stores state relevant to thresholding processes.
    """

    def __init__(self) -> None:
        super().__init__()

        # cyto-dl segmentations should have values between 0 and 255
        self._thresholding_value_selected: int = 120
        self._autothresholding:bool = False


    def set_thresholding_value(self, value: int) -> None:
        """
        Set the thresholding value.
        """
        self._thresholding_value_selected = value
        self.dispatch(Event.ACTION_THRESHOLDING_VALUE_CHANGED)


    def get_thresholding_value(self) -> int:
        """
        Get the thresholding value.
        """
        return self._thresholding_value_selected

    def set_autothresholding_enabled(self) -> None:
        """
        Set autothresholding enabled.
        """
        self._autothresholding = True
        self.dispatch(Event.ACTION_THRESHOLDING_AUTOTHRESHOLDING_SELECTED)

