from allencell_ml_segmenter.model.event import MainEvent
from enum import Enum
from allencell_ml_segmenter.model.publisher import Publisher


class Page(Enum):
    """
    Different pages in the UI
    """

    MAIN = "main"
    TRAINING = "training"
    PREDICTION = "prediction"
    CURATION = "curation"


class MainModel(Publisher):
    """
    Main model for this application
    """

    def __init__(self):
        super().__init__()
        # Current page of the UI
        self._current_page: Page = Page.MAIN

    @property
    def current_page(self) -> bool:
        """
        getter/property for current page
        """
        return self._current_page

    def set_current_page(self, page: Page):
        """
        Set the current page in the UI and dispatch a MainEvent
        """
        self._current_page = page
        self.dispatch(MainEvent(page.value))
