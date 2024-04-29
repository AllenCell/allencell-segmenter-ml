from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class MainModel(Publisher):
    """
    Main model for this application. Responsible for non-tab-related view switching.
    """

    def __init__(self):
        super().__init__()
        # Current page of the UI
        self._current_view: AicsWidget = None
        self._is_new_model: bool = False

    def get_current_view(self):
        """
        getter/property for current page
        """
        return self._current_view

    def set_current_view(self, view):
        """
        Set the current page in the UI and dispatch a MainEvent
        """
        self._current_view = view
        self.dispatch(Event.ACTION_CHANGE_VIEW)

    def set_new_model(self, is_new_model: bool):
        """
        Dispatches a new model event
        """
        self._is_new_model = is_new_model
        self.dispatch(Event.ACTION_NEW_MODEL)

    def is_new_model(self):
        """
        getter/property for is_new_model
        """
        return self._is_new_model