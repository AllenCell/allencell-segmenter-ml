from pathlib import Path
import pytest
import responses

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_user_settings import (
    FakeUserSettings,
)
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.utils.s3.s3_bucket_constants import PROD_BUCKET
from allencell_ml_segmenter.widgets.model_download_dialog import (
    ModelDownloadDialog,
)
from allencell_ml_segmenter._tests.utils.s3.s3_response_fixtures import (
    s3_response_listobjectv2_contents_two_models,
)


@pytest.fixture
@responses.activate
def model_download_dialog(
    s3_response_listobjectv2_contents_two_models: str,
) -> ModelDownloadDialog:
    fake_settings: FakeUserSettings = FakeUserSettings()
    fake_settings.set_user_experiments_path(
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "output_test_folder"
    )
    exp_model: ExperimentsModel = ExperimentsModel(fake_settings)

    # add fake xml response that is returned when we make a request to the test_url with the list-type=2 param
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{PROD_BUCKET}?list-type=2",
            "body": s3_response_listobjectv2_contents_two_models,
            "status": 200,
            "content_type": "application/xml",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )
    return ModelDownloadDialog(None, exp_model)


def test_model_download_dialog_init(
    model_download_dialog: ModelDownloadDialog,
):
    # ASSERT
    assert (
        model_download_dialog._model_select_dropdown.count() == 2
    )  # two items added to combobox
