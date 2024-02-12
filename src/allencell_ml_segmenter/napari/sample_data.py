"""
This allows us to provide users with sample data for testing the plugin
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""

from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np


def make_sample_data() -> List[Tuple[np.ndarray, Dict]]:
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image
    return [(np.random.rand(512, 512), {})]
