from enum import Enum


class Event(Enum):
    """
    Different application Events
    """

    TRAINING = "training"
    PREDICTION = "prediction"
    MAIN = "main"
    CHANGE_VIEW = "change_view"
