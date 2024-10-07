from allencell_ml_segmenter.main.i_experiments_model import IExperimentsModel
from pathlib import Path
from typing import Optional


class ExperimentUtils:
    """
    ExperimentUtils handles experiment navigation.
    """

    @staticmethod
    def get_best_ckpt(user_exp_path: Path, experiment: str) -> Optional[Path]:
        checkpoints_path = user_exp_path / experiment / "checkpoints"
        if not checkpoints_path.exists():
            return None

        files: list[Path] = [
            entry
            for entry in checkpoints_path.iterdir()
            if entry.is_file() and not "last" in entry.name.lower()
        ]
        if not files:
            return None

        files.sort(key=lambda file: file.stat().st_mtime)
        return checkpoints_path / files[-1].name
