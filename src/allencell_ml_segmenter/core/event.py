from enum import Enum


class Event(Enum):
    """
    Different application Events
    """

    # Process events
    TRAINING = "training"
    PREDICTION = "prediction"
    
    # View selection events
    MAIN = "main"
    CHANGE_VIEW = "change_view"
    TRAINING_SELECTED = "training_selected"
    PREDICTION_SELECTED = "prediction_selected"
    MAIN_SELECTED = "main_selected"
