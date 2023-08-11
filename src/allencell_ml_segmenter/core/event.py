from enum import Enum


class Event(Enum):
    """
    Different application Events
    """

    # Process events. This signals that a long-running process is active. Progress updates should be shown to the user.
    PROCESS_TRAINING = "training"
    PROCESS_TRAINING_PROGRESS = "training_progress"
    PROCESS_TRAINING_SHOW_ERROR = "training_error"
    PROCESS_TRAINING_CLEAR_ERROR = "training_clear_error"
    PROCESS_TRAINING_COMPLETE = "training_complete"
    PROCESS_PREDICTION = "prediction"

    # Action events. This signals a change in the UI. These are a direct result of a user action
    ACTION_CHANGE_VIEW = "change_view"
    ACTION_START_TRAINING = "start_training"
    ACTION_PREDICTION_MODEL_FILE = "model_file"
    ACTION_PREDICTION_PREPROCESSING_METHOD = "preprocessing_method"
    ACTION_PREDICTION_POSTPROCESSING_METHOD = "postprocessing_method"
    ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD = (
        "postprocessing_simple_threshold"
    )
    ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD = (
        "postprocessing_auto_threshold"
    )

    # View selection events. These can stem from a user action, or from a process (i.e. prediction process ends, and a new view is shown automatically).
    VIEW_SELECTION_TRAINING = "training_selected"
    VIEW_SELECTION_PREDICTION = "prediction_selected"
