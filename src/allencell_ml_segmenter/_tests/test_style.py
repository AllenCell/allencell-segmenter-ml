import pytest

from allencell_ml_segmenter._style import Style


class TestStyle:
    @pytest.mark.parametrize("name", [None, "not_a_qt_stylesheet.txt"])
    def test_load_stylesheet_bad_name(self, name: str) -> None:
        with pytest.raises(ValueError):
            Style.get_stylesheet(name)

    def test_load_stylesheet_bad_path(self) -> None:
        with pytest.raises(IOError):
            Style.get_stylesheet("not_found.qss")

    def test_load_stylesheet(self) -> None:
        contents = Style.get_stylesheet("prediction_view.qss")
        assert contents is not None
        assert len(contents) > 0
