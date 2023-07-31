import allencell_ml_segmenter

from pathlib import Path


class Directories:
    """
    Provides safe paths to common module directories
    """

    _module_base_dir: Path = Path(allencell_ml_segmenter.__file__).parent

    @classmethod
    def get_assets_dir(cls) -> str:
        """
        Path to the assets directory
        """
        return str(cls._module_base_dir / "assets")

    @classmethod
    def get_style_dir(cls) -> Path:
        """
        Path to the stylesheet directory
        """
        return cls._module_base_dir / "styles"
