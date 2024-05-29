from pathlib import Path
from typing import List
from unittest.mock import Mock, mock_open, patch, call

import numpy as np
import pytest
from napari.layers import Shapes

from allencell_ml_segmenter._tests.fakes.fake_subscriber import FakeSubscriber
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer
from allencell_ml_segmenter.core.event import Event
from allencell_ml_segmenter.curation.curation_data_class import CurationRecord
from allencell_ml_segmenter.curation.curation_model import CurationModel
from allencell_ml_segmenter.curation.curation_service import (
    CurationService,
)
from allencell_ml_segmenter.core.image_data_extractor import ImageData
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.main.viewer import Viewer
import allencell_ml_segmenter


"""
Getting the curation service tests to work will require some significant changes, which will impact
modules outside of curation. I think it's wise to defer that to a later PR. Here are the important
changes that need to happen to unit test curation service.

1. have curation service use TaskExecutor instead of @thread_worker annotations
    - this is a change that will not impact things outside of curation
2. change FileUtils so that it is a singleton with a global instance instead of a class with
static methods. This will allow us to use dependency injection for testing.
3. create a fake for FileUtils that saves things to RAM instead of disk (similar to our fake
viewer)
"""
