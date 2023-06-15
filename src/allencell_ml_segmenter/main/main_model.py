from allencell_ml_segmenter.views.view import View
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher


class MainModel(Publisher):
    """
    Main models for this application
    """

    def __init__(self):
        super().__init__()
        # Current page of the UI
        self._current_view: View = None
        self.training_running: bool = False
        self.perdictions_running: bool = False

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
        self.dispatch(Event.ACTION_CHANGE_VIEW)

    def get_training_running(self) -> bool:
        """
        getter/property for training running
        """
        return self.training_running
    
    def set_training_running(self, training_running: bool):     
        """
        Set the training running in the UI and dispatch a MainEvent
        """
        self.training_running = training_running
        self.dispatch(Event.PROCESS_TRAINING)   

    def get_perdictions_running(self) -> bool:   
        """
        getter/property for perdictions running
        """
        return self.perdictions_running
    
    def set_perdictions_running(self, perdictions_running: bool):   
        """
        Set the perdictions running in the UI and dispatch a MainEvent
        """
        self.perdictions_running = perdictions_running
        self.dispatch(Event.PREDICTIONS)
