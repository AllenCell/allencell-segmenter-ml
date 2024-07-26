from pathlib import Path
from allencell_ml_segmenter.core.subscriber import Subscriber
from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from typing import Dict, Callable, List, Optional
from allencell_ml_segmenter.core.event import Event
import allencell_ml_segmenter

test_channel_path: Path = (
    Path(allencell_ml_segmenter.__file__).parent
    / "_tests"
    / "test_files"
    / "channel_selection_json"
    / "valid_0.json"
)


class FakeExperimentsModel(IExperimentsModel):
    def __init__(
        self,
        experiments: List[str] = ["0_exp", "1_exp", "2_exp"],
        channel_selection_path: Path = test_channel_path,
    ):
        super().__init__()
        self._experiments = experiments
        self._experiment_name = None
        self._checkpoint = None
        self._events_to_subscriber_handlers: Dict[
            Dict[Subscriber, Callable]
        ] = {event: dict() for event in [e.value for e in Event]}
        self._chan_sel_path = channel_selection_path

    def get_checkpoint(self) -> str:
        return self._checkpoint

    def set_checkpoint(self, checkpoint: str):
        self._checkpoint = checkpoint

    def get_experiments(self) -> List[str]:
        return self._experiments

    def refresh_experiments(self):
        pass

    def refresh_checkpoints(self, experiment: str):
        pass

    def get_user_experiments_path(self) -> Path:
        return Path()

    def get_model_checkpoints_path(
        self, experiment_name: str, checkpoint: str
    ) -> Path:
        return Path()

    def get_metrics_csv_path(self) -> Path:
        return Path()

    def get_latest_metrics_csv_version(self) -> int:
        pass

    def get_latest_metrics_csv_path(self) -> Optional[Path]:
        return Path()

    def get_csv_path(self) -> Path:
        return Path()

    def get_cache_dir(self) -> Path:
        return Path()

    def get_channel_selection_path(self) -> Optional[Path]:
        return self._chan_sel_path
