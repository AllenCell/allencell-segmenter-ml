import pytest

from allencell_ml_segmenter._style import Style


class TestStyle:
    @pytest.mark.parametrize("name", [None, "not_a_qt_stylesheet.txt"])
    def test_load_stylesheet_bad_name(self, name: str) -> None:
        """
        Tests that an error is raised when trying to load a stylesheet with an invalid name.
        """
        with pytest.raises(ValueError):
            Style.get_stylesheet(name)

    def test_load_stylesheet_bad_path(self) -> None:
        """
        Tests that an error is raised when trying to load a stylesheet that doesn't exist.
        """
        with pytest.raises(IOError):
            Style.get_stylesheet("not_found.qss")

    def test_load_stylesheet(self) -> None:
        """
        Tests that an existing stylesheet is properly loaded.
        """
        contents = Style.get_stylesheet("prediction_view.qss")
        assert contents is not None
        assert len(contents) > 0
