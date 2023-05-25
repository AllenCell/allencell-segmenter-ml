from allencell_ml_segmenter.model.event import MainEvent
from enum import Enum
from allencell_ml_segmenter.model.publisher import Publisher

class Page(Enum):
    MAIN = "main"
    TRAINING = "training"
    PREDICTION = "prediction"
    CURATION = "curation"

class MainModel(Publisher):
    def __init__(self):
        super().__init__()
        self._current_page: Page = Page.MAIN

    @property
    def current_page(self) -> bool:
        return self._current_page

    def set_current_page(self, page: Page):
        self._current_page = page
        self.dispatch(MainEvent(page.value))

