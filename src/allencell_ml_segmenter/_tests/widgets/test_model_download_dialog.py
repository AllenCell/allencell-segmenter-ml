from pathlib import Path
import pytest
import responses

import allencell_ml_segmenter
from allencell_ml_segmenter._tests.fakes.fake_user_settings import FakeUserSettings
from allencell_ml_segmenter.main.experiments_model import ExperimentsModel
from allencell_ml_segmenter.widgets.model_download_dialog import ModelDownloadDialog

@pytest.fixture
@responses.activate
def model_download_dialog() -> ModelDownloadDialog:
    fake_settings: FakeUserSettings = FakeUserSettings()
    fake_settings.set_user_experiments_path(
        Path(allencell_ml_segmenter.__file__).parent / "_tests" / "test_files" / "output_test_folder")
    exp_model: ExperimentsModel = ExperimentsModel(fake_settings)
    fake_url: str = "http://fakeurl.com"
    # The following lines mimic what s3 would send back from the ListObjectV2 http request
    # example response is from https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    # and does not represent any real data.
    xml_response: str = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f"<ListBucketResult>"
        f"<Name>bucket</Name>"
        f"<Prefix></Prefix>"
        f"<ContinuationToken>randomtoken</ContinuationToken>"
        f"<KeyCount>112</KeyCount>"
        f"<MaxKeys>1000</MaxKeys>"
        f"<IsTruncated>false</IsTruncated>"
        f"  <Contents>"
        f"    <Key>model1.zip</Key>"  # 1st Model file name
        f"    <LastModified>2014-11-21T19:40:05.000Z</LastModified>"
        f'    <ETag>"70ee1738b6b21e2c8a43f3a5ab0eee71"</ETag>'
        f"    <Size>1111</Size>"
        f"    <StorageClass>STANDARD</StorageClass>"
        f"  </Contents>"
        f"  <Contents>"
        f"    <Key>model2.zip</Key>"  # 2nd Model file name (duplicated)
        f"    <LastModified>2014-11-21T19:40:05.000Z</LastModified>"
        f'     <ETag>"70ee1738b6b21e2c8a43f3a5ab0eee71"</ETag>'
        f"    <Size>1111</Size>"
        f"    <StorageClass>STANDARD</StorageClass>"
        f"  </Contents>"
        f"  <Contents>"
        f"    <Key>some_random_file</Key>"  # some random file
        f"    <LastModified>2014-11-21T19:40:05.000Z</LastModified>"
        f'    <ETag>"70ee1738b6b21e2c8a43f3a5ab0eee71"</ETag>'
        f"    <Size>1111</Size>"
        f"    <StorageClass>STANDARD</StorageClass>"
        f"  </Contents>"
        f"</ListBucketResult>"
    )
    # add fake xml response that is returned when we make a request to the test_url with the list-type=2 param
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{fake_url}?list-type=2",
            "body": xml_response,
            "status": 200,
            "content_type": "application/xml",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )
    return ModelDownloadDialog(None, exp_model, test_s3_bucket=fake_url)

def test_model_download_dialog_init(model_download_dialog: ModelDownloadDialog):
    # ASSERT
    assert model_download_dialog._model_select_dropdown.count() == 2 # two items added to combobox

