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
    PROCESS_PREDICTION_COMPLETE = "prediction_complete"
    PROCESS_CURATION_INPUT_STARTED = "curation_input_started"
    PROCESS_CURATION_NEXT_IMAGE = "curation_next_image"

    # Action events. This signals a change in the UI. These are a direct result of a user action
    ACTION_REFRESH = "refresh"
    ACTION_CHANGE_VIEW = "change_view"
    ACTION_NEW_MODEL = "new_model"
    # Training
    ACTION_START_TRAINING = "start_training"
    ACTION_TRAINING_MAX_NUMBER_CHANNELS_SET = "max_number_channel_set_training"
    ACTION_TRAINING_DATASET_SELECTED = "training_dataset_selected"
    # Prediction
    ACTION_PREDICTION_MODEL_FILE = "model_file"
    ACTION_PREDICTION_PREPROCESSING_METHOD = "preprocessing_method"
    ACTION_PREDICTION_POSTPROCESSING_METHOD = "postprocessing_method"
    ACTION_PREDICTION_POSTPROCESSING_SIMPLE_THRESHOLD = (
        "postprocessing_simple_threshold"
    )
    ACTION_PREDICTION_POSTPROCESSING_AUTO_THRESHOLD = (
        "postprocessing_auto_threshold"
    )
    ACTION_PREDICTION_GET_IMAGE_PATHS_FROM_NAPARI = (
        "prediction_get_image_paths_from_napari"
    )
    ACTION_PREDICTION_MAX_CHANNELS_SET = "prediction_max_channels_set"
    ACTION_PREDICTION_SETUP = "prediction_setup"
    ACTION_PREDICTION_EXTRACT_CHANNELS = "prediction_extract_channels"

    # Curation
    ACTION_CURATION_RAW_CHANNELS_SET = "curation_raw_channels_set"
    ACTION_CURATION_SEG1_CHANNELS_SET = "curation_seg1_channels_set"
    ACTION_CURATION_SEG2_CHANNELS_SET = "curation_seg2_channels_set"
    # Experiment related
    ACTION_EXPERIMENT_SELECTED = "experiment_selected"
    ACTION_EXPERIMENT_APPLIED = "experiment_applied"
    ACTION_CURATION_DRAW_EXCLUDING = "curation_draw_excluding"
    ACTION_CURATION_FINISHED_DRAW_EXCLUDING = (
        "curation_finished_draw_excluding"
    )
    ACTION_CURATION_DRAW_MERGING = "curation_draw_merging"
    ACTION_CURATION_SAVE_EXCLUDING_MASK = "curation_save_excluding_mask"
    ACTION_CURATION_SAVED_MERGING_MASK = "curation_saved_merging_mask"
    # error handling
    ACTION_CURATION_SEG2_THREAD_ERROR = "curation_seg2_thread_error"
    ACTION_CURATION_SEG1_THREAD_ERROR = "curation_seg1_thread_error"
    ACTION_CURATION_RAW_THREAD_ERROR = "curation_raw_thread_error"

    # View selection events. These can stem from a user action, or from a process (i.e. prediction process ends, and a new view is shown automatically).
    VIEW_SELECTION_TRAINING = "training_selected"
    VIEW_SELECTION_PREDICTION = "prediction_selected"
