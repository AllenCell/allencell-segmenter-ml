import pytest
from pathlib import Path
import numpy as np

from allencell_ml_segmenter.core.file_input_model import FileInputModel, InputMode
from allencell_ml_segmenter._tests.fakes.fake_experiments_model import FakeExperimentsModel
from allencell_ml_segmenter.main.main_model import MainModel
from allencell_ml_segmenter.thresholding.thresholding_model import ThresholdingModel
from allencell_ml_segmenter.thresholding.thresholding_service import ThresholdingService
from allencell_ml_segmenter.core.task_executor import SynchroTaskExecutor
from allencell_ml_segmenter._tests.fakes.fake_viewer import FakeViewer


@pytest.fixture
def test_image():
    """Create a small test image for thresholding."""
    return np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

@pytest.fixture
def thresholding_model():
    return ThresholdingModel()

@pytest.fixture
def viewer():
    return FakeViewer()


@pytest.fixture
def thresholding_service(thresholding_model, viewer):
    """Fixture to initialize the ThresholdingService with SynchroTaskExecutor and FakeViewer."""
    return ThresholdingService(
        thresholding_model,
        FakeExperimentsModel(),
        FileInputModel(),
        MainModel(),
        viewer,
        task_executor=SynchroTaskExecutor.global_instance(),
    )

def test_on_threshold_changed_non_prediction(test_image):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    viewer: FakeViewer = FakeViewer()
    thresholding_service: ThresholdingService = ThresholdingService(thresholding_model, FakeExperimentsModel(),
                                                                    FileInputModel(), MainModel(), viewer,
                                                                    task_executor=SynchroTaskExecutor.global_instance())
    viewer.add_image(test_image, name="test_layer")

    # ACT set a threshold to trigger
    thresholding_model.set_thresholding_value(50)

    # Verify a segmentation layer is added
    assert "[threshold] test_layer" in viewer.segmentation_inserted
    seg_data = viewer.segmentation_inserted["[threshold] test_layer"]
    assert np.array_equal(seg_data, (test_image > 50).astype(int))

    # check if existing thresholds get updated
    thresholding_model.set_thresholding_value(100)
    assert len(viewer.get_layers()) == 1
    seg_data = viewer.segmentation_inserted["[threshold] test_layer"]
    assert np.array_equal(seg_data, (test_image > 100).astype(int))

def test_on_threshold_changed_non_prediction(test_image):
    thresholding_model: ThresholdingModel = ThresholdingModel()
    viewer: FakeViewer = FakeViewer()
    main_model: MainModel = MainModel()
    main_model.set_predictions_in_viewer(True)
    thresholding_service: ThresholdingService = ThresholdingService(thresholding_model, FakeExperimentsModel(),
                                                                    FileInputModel(), main_model, viewer,
                                                                    task_executor=SynchroTaskExecutor.global_instance())
    viewer.add_image(test_image, name="[raw] test_layer 1")
    viewer.add_image(test_image, name="[seg] test_layer 1")
    viewer.add_image(test_image, name="[raw] test_layer 2")
    viewer.add_image(test_image, name="[seg] test_layer 2")
    viewer.add_image(test_image, name="donotthreshold")

    # ACT set a threshold to trigger
    thresholding_model.set_thresholding_value(50)

    # Verify a threshold layer is added for each seg layer
    assert "[threshold] [seg] test_layer 1" in viewer.segmentation_inserted
    seg_data = viewer.segmentation_inserted["[threshold] [seg] test_layer 1"]
    assert np.array_equal(seg_data, (test_image > 50).astype(int))
    assert "[threshold] [seg] test_layer 2" in viewer.segmentation_inserted
    seg_data = viewer.segmentation_inserted["[threshold] [seg] test_layer 2"]
    assert np.array_equal(seg_data, (test_image > 50).astype(int))
    # verify that raw layers do not get thresholded
    assert len(viewer.segmentation_inserted) == 2

    # verify existing threshold layers get updated correctly
    thresholding_model.set_thresholding_value(100)
    # Verify a threshold layer is added for each seg layer
    assert "[threshold] [seg] test_layer 1" in viewer.segmentation_inserted
    seg_data = viewer.segmentation_inserted["[threshold] [seg] test_layer 1"]
    assert np.array_equal(seg_data, (test_image > 100).astype(int))
    assert "[threshold] [seg] test_layer 2" in viewer.segmentation_inserted
    seg_data = viewer.segmentation_inserted["[threshold] [seg] test_layer 2"]
    assert np.array_equal(seg_data, (test_image > 100).astype(int))
    # verify that raw layers do not get thresholded
    assert len(viewer.segmentation_inserted) == 2


# def test_save_thresholded_images(thresholding_service, setup_models_and_viewer, test_image, tmp_path):
#     """
#     Test the _save_thresholded_images method saves thresholded images correctly.
#     """
#     _, _, file_input_model, thresholding_model, viewer = setup_models_and_viewer
#
#     # Prepare input and output paths
#     input_file = tmp_path / "test_image.tiff"
#     output_dir = tmp_path / "output"
#     output_dir.mkdir()
#
#     file_input_model.set_input_mode(InputMode.FROM_PATH)
#     file_input_model.set_input_image_path(input_file)
#     file_input_model.set_output_directory(output_dir)
#
#     # Write test image to input file
#     from bioio.writers import OmeTiffWriter
#     OmeTiffWriter.save(test_image, str(input_file))
#
#     # Set threshold and save
#     thresholding_model.set_thresholding_value(50)
#     thresholding_service._save_thresholded_images(Event.ACTION_SAVE_THRESHOLDING_IMAGES)
#
#     # Verify output file exists
#     output_file = output_dir / "threshold_test_image.tiff"
#     assert output_file.exists()
#
#     # Verify contents of output file
#     from bioio import BioImage
#     saved_image = BioImage(output_file).data
#     assert np.array_equal(saved_image, (test_image > 50).astype(int))
#
#
# def test_threshold_image(thresholding_service, test_image):
#     """
#     Test the _threshold_image method applies a threshold correctly.
#     """
#     thresholding_service._thresholding_model.set_thresholding_value(50)
#     thresholded_image = thresholding_service._threshold_image(test_image)
#     assert np.array_equal(thresholded_image, (test_image > 50).astype(int))
