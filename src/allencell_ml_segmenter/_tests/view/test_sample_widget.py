import pytest
from allencell_ml_segmenter.view.sample_widget import SampleWidget

@pytest.fixture
def sample_widget(qtbot):
    return SampleWidget()

def test_set_label_text(sample_widget):
    text = "Training in progress"
    sample_widget.setLabelText(text)
    assert sample_widget.label.text() == text

def test_connect_slots(sample_widget):
    function_called = False
    def callable_test():
        # nonlocal variable available outside function
        nonlocal function_called
        function_called = True

    sample_widget.connectSlots(callable_test)
    sample_widget.btn.click()

    assert function_called

