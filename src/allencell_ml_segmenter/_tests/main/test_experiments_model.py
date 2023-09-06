import pytest
from allencell_ml_segmenter.config.cyto_dl_config import CytoDlConfig
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel


@pytest.fixture
def config() -> CytoDlConfig:
    """
    Fixture for MainModel testing.
    """
    return CytoDlConfig()


def refresh_experiments(self) -> None:
    model = ExperimentsModel()
    experimentsModel = model.refresh_experiments()
    assert experimentsModel is not None
