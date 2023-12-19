# from pathlib import Path
from pathlib import PurePath
import pytest
from allencell_ml_segmenter.config.user_config import CytoDlConfig

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel


@pytest.fixture
def config() -> CytoDlConfig:
    """
    Fixture for MainModel testing.
    """
    return CytoDlConfig()


def test_refresh_experiments() -> None:
    model = ExperimentsModel(
        CytoDlConfig(
            cyto_dl_home_path=PurePath(__file__).parent / "cyto_dl_home",
            user_experiments_path=PurePath(__file__).parent
            / "experiments_home",
        )
    )
    expected = {"0_exp": set(), "1_exp": set(), "2_exp": {"0.ckpt", "1.ckpt"}}
    assert model.get_experiments() == expected


def test_get_cyto_dl_config() -> None:
    expected_config = CytoDlConfig(
        cyto_dl_home_path=PurePath(__file__).parent / "cyto_dl_home",
        user_experiments_path=PurePath(__file__).parent / "experiments_home",
    )
    model = ExperimentsModel(expected_config)
    assert model.get_cyto_dl_config() == expected_config


def test_get_user_experiments_path() -> None:
    user_experiments_path = PurePath(__file__).parent / "experiments_home"
    config = CytoDlConfig(
        cyto_dl_home_path=PurePath(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    model = ExperimentsModel(config)
    assert model.get_user_experiments_path() == user_experiments_path


def test_get_model_checkpoints() -> None:
    user_experiments_path = PurePath(__file__).parent / "experiments_home"
    config = CytoDlConfig(
        cyto_dl_home_path=PurePath(__file__).parent / "cyto_dl_home",
        user_experiments_path=user_experiments_path,
    )
    expected = user_experiments_path / "foo" / "checkpoints" / "bar"
    model = ExperimentsModel(config)
    assert model.get_model_checkpoints_path("foo", "bar") == expected
