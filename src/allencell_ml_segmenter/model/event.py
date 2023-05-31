from enum import Enum


class Event(Enum):
    """
    Different application Events
    """

    TRAINING = "training"


class MainEvent(Enum):
    """
    Events that determine which page to show.
    """

    MAIN = "main"
    TRAINING = "training"
    PREDICTION = "prediction"
    CURATION = "curation"
