from allencell_ml_segmenter.view.view import View
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher

class MainModel(Publisher):
    """
    Main model for this application
    """

    def __init__(self):
        super().__init__()
        # Current page of the UI
        self._current_view: View = None

    def get_current_view(self) -> bool:
        """
        getter/property for current page
        """
        return self._current_view

    def set_current_view(self, view: View):
        """
        Set the current page in the UI and dispatch a MainEvent
        """
        self._current_view = view
        self.dispatch(Event.CHANGE_VIEW)
