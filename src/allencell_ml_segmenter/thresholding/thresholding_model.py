from typing import Optional
from napari.layers import Layer  # type: ignore

from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.file_input_model import FileInputModel

# Some thresholding constants #
AVAILABLE_AUTOTHRESHOLD_METHODS: list[str] = ["threshold_otsu"]
THRESHOLD_DEFAULT = 120
THRESHOLD_RANGE = (0, 255)


class ThresholdingModel(FileInputModel):
    """
    Stores state relevant to thresholding processes.
    """

    def __init__(self) -> None:
        super().__init__()

        # cyto-dl segmentations should have values between 0 and 255
        self._is_threshold_enabled: bool = False
        self._thresholding_value_selected: int = THRESHOLD_DEFAULT
        self._is_autothresholding_enabled: bool = False
        self._autothresholding_method: str = AVAILABLE_AUTOTHRESHOLD_METHODS[0]
        self._thresholding_layers: list[Layer] = (
            []
        )  # Layers show binary map as data, but contain a metadata
        # key prob_map which contains the probability map that we need to threshold to generate a binary map with
        # the set threshold value.
        self._selected_idx: list[int] = []

    def set_thresholding_value(self, value: int) -> None:
        """
        Set the thresholding value.
        """
        self._thresholding_value_selected = value
        self.dispatch(Event.ACTION_EXECUTE_THRESHOLDING)

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
        if enable:
            self.dispatch(Event.ACTION_EXECUTE_THRESHOLDING)

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
        self.dispatch(Event.ACTION_EXECUTE_THRESHOLDING)

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
        if enabled:
            self.dispatch(Event.ACTION_EXECUTE_THRESHOLDING)

    def is_threshold_enabled(self) -> bool:
        """
        Get threshold specific value.
        """
        return self._is_threshold_enabled

    def disable_all(self) -> None:
        self.set_threshold_enabled(False)
        self.set_autothresholding_enabled(False)
        self.dispatch(Event.ACTION_THRESHOLDING_DISABLED)

    def dispatch_save_thresholded_images(self) -> None:
        self.dispatch(Event.ACTION_SAVE_THRESHOLDING_IMAGES)

    def set_thresholding_layers(self, layers: list[Layer]) -> None:
        self._thresholding_layers = layers

    def get_thresholding_layers(self) -> list[Layer]:
        return self._thresholding_layers

    def set_selected_idx(self, selected_idx: list[int]) -> None:
        self._selected_idx = selected_idx
        self.dispatch(Event.ACTION_EXECUTE_THRESHOLDING)

    def get_selected_idx(self) -> list[int]:
        return self._selected_idx
