from enum import Enum


class Event(Enum):
    TRAINING = "training"


class MainEvent(Enum):
    MAIN = "main"
    TRAINING = "training"
    PREDICTION = "prediction"
    CURATION = "curation"
    