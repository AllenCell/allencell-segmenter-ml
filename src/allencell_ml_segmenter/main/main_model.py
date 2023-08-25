from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
from allencell_ml_segmenter.core.aics_widget import AicsWidget
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.core.publisher import Publisher
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel


class MainModel(Publisher):
    """
    Main model for this application. Responsible for non-tab-related view switching.
    """

    def __init__(self, experiments_model: ExperimentsModel):
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

    def get_experiment_model(self) -> ExperimentsModel:
        return self._experiments_model
