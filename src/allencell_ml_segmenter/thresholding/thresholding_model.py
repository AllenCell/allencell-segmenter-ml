from typing import Optional
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher

AVAILABLE_AUTOTHRESHOLD_METHODS: list[str] = ["threshold_otsu"]


class ThresholdingModel(Publisher):
    """
    Stores state relevant to thresholding processes.
    """

    def __init__(self) -> None:
        super().__init__()

        # cyto-dl segmentations should have values between 0 and 255
        self._is_threshold_enabled: bool = False
        self._thresholding_value_selected: int = 120
        self._is_autothresholding_enabled: bool = False
        self._autothresholding_method: Optional[str] = AVAILABLE_AUTOTHRESHOLD_METHODS[0]


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

    def set_autothresholding_enabled(self, enable: bool) -> None:
        """
        Set autothresholding enabled.
        """
        self._is_autothresholding_enabled = enable
        self.dispatch(Event.ACTION_THRESHOLDING_AUTOTHRESHOLDING_SELECTED)

    def is_autothresholding_enabled(self) -> bool:
        """
        Get autothresholding enabled.
        """
        return self._is_autothresholding_enabled

    def set_autothresholding_method(self, method: str) -> None:
        """
        Set autothresholding method.
        """
        self._autothresholding_method = method
        self.dispatch(Event.ACTION_THRESHOLDING_AUTOTHRESHOLDING_SELECTED)

    def get_autothresholding_method(self) -> str:
        """
        Get autothresholding method.
        """
        return self._autothresholding_method

    def set_threshold_enabled(self, enabled: bool) -> None:
        """
        Set threshold specific value.
        """
        self._is_threshold_enabled = enabled

    def is_threshold_enabled(self) -> bool:
        """
        Get threshold specific value.
        """
        return self._is_threshold_enabled

    def dispatch_save_thresholded_images(self) -> None:
        self.dispatch(Event.ACTION_SAVE_THRESHOLDING_IMAGES)

