from pathlib import Path


class UserConfig:
    def __init__(self, cyto_dl_home_path: Path, user_experiments_path: Path):
        self._cyto_dl_home_path: Path = Path(cyto_dl_home_path)
        self._user_experiments_path: Path = Path(user_experiments_path)

    def get_cyto_dl_home_path(self) -> Path:
        return self._cyto_dl_home_path

    def get_user_experiments_path(self) -> Path:
        return self._user_experiments_path
