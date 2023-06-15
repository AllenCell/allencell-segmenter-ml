from enum import Enum


class Event(Enum):
    """
    Different application Events
    """

    # Process events.  This signals long running process is active.  Porgress updates should be shown to the user. 
    PROCESS_TRAINING = "training"
    PROCESS_PREDICTION = "prediction"
    
    # Action events.  This signals a change in the UI.  These are a direct result of a user action
    ACTION_CHANGE_VIEW = "change_view"

    # View selection events.  These can stem from a user action, or from a process (ie prediction process ends, and a new view is shown automatically).
    VIEW_SELECTION_TRAINING = "training_selected"
    VIEW_SELECTION_PREDICTION = "prediction_selected"
    VIEW_SELECTION_MAIN = "main_selected"
