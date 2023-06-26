import pytest
from qtpy.QtWidgets import QPushButton
from unittest.mock import Mock
from allencell_ml_segmenter.sample.sample_state_widget import SampleStateWidget
from allencell_ml_segmenter.main.main_model import MainModel


@pytest.fixture
def sample_widget(qtbot):
    return SampleStateWidget(MainModel())


# def test_init(sample_widget):
# assert isinstance(sample_widget.btn, QPushButton)
# assert sample_widget.btn.text() == "Start Training"
# assert isinstance(sample_widget.return_btn, QPushButton)
# assert sample_widget.return_btn.text() == "Return"
