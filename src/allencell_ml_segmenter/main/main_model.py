from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel


class MainModel(Publisher):
    """
    Main model for this application. Responsible for non-tab-related view switching.
    """

    def __init__(self, experiments_model: IExperimentsModel):
        super().__init__()
        # Current page of the UI
        self._current_view: AicsWidget = None
        self._experiments_model = experiments_model

    def get_current_view(self) -> AicsWidget:
        """
        getter/property for current page
        """
        return self._current_view

    def set_current_view(self, view: AicsWidget):
        """
        Set the current page in the UI and dispatch a MainEvent
        """
        self._current_view = view
        self.dispatch(Event.ACTION_CHANGE_VIEW)

    def get_experiment_model(self) -> IExperimentsModel:
        return self._experiments_model
