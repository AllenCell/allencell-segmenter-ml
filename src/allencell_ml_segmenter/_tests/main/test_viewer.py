from napari.layers import Labels
import numpy as np

from allencell_ml_segmenter.main.viewer import Viewer


def test_clear_binary_map_from_layer() -> None:
    layer: Labels = Labels(np.ones((10, 10, 10), dtype=bool))

    Viewer.clear_binary_map_from_layer(Viewer, layer)

    assert np.all(layer.data == 0)  # assert all values are now zeros
