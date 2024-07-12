from pathlib import Path

import pytest
import responses

import allencell_ml_segmenter
from allencell_ml_segmenter.utils.s3.s3_available_model import AvailableModel
from allencell_ml_segmenter.utils.s3.s3_model_bucket import (
    S3ModelBucket,
)
from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)
from allencell_ml_segmenter._tests.utils.s3.s3_response_fixtures import (
    s3_response_listobjectv2_contents_two_models,
    s3_response_listobjectv2_contents_duplicate_model,
)
from allencell_ml_segmenter.utils.s3.s3_bucket_constants import PROD_BUCKET


@responses.activate
def test_get_available_models(
    s3_response_listobjectv2_contents_two_models: str,
) -> None:
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
    # Test path to save zip to
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
    )

    model_downloader: S3ModelBucket = S3ModelBucket(PROD_BUCKET, test_path)

    # ACT
    available_models_dict: dict[str, AvailableModel] = (
        model_downloader.get_available_models()
    )

    # ASSERT
    # should have 2 models available
    assert len(available_models_dict.keys()) == 2
    assert "model1" in available_models_dict.keys()
    assert "model2" in available_models_dict.keys()
    assert "some_random_file" not in available_models_dict.keys()

    # dictionary values contain available models object w/ name and object url on s3
    # check if object url was built correctly for the AvailableModels
    assert available_models_dict["model1"].get_name() == "model1.zip"
    assert (
        available_models_dict["model1"].get_object_url()
        == f"{PROD_BUCKET}/model1.zip"
    )
    assert available_models_dict["model2"].get_name() == "model2.zip"
    assert (
        available_models_dict["model2"].get_object_url()
        == f"{PROD_BUCKET}/model2.zip"
    )


@responses.activate
def test_get_available_models_duplicate_file_error(
    s3_response_listobjectv2_contents_duplicate_model: str,
) -> None:
    # ARRANGE
    # add fake xml response that is returned when we make a request to the test_url with the list-type=2 param
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{PROD_BUCKET}?list-type=2",
            "body": s3_response_listobjectv2_contents_duplicate_model,
            "status": 200,
            "content_type": "application/xml",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )
    # Test path to save zip to
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
    )

    model_downloader: S3ModelBucket = S3ModelBucket(PROD_BUCKET, test_path)

    # ACT/ASSERT
    with pytest.raises(ValueError):
        # if there are models with duplicate names on s3 somehow- we show an error
        model_downloader.get_available_models()


@responses.activate
def test_get_available_models_bad_request() -> None:
    # ARRANGE
    error_response: str = "error"
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{PROD_BUCKET}?list-type=2",
            "body": error_response,
            "status": 400,
            "content_type": "error",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )
    # Test path to save zip to
    test_path: Path = (
        Path(allencell_ml_segmenter.__file__).parent
        / "_tests"
        / "test_files"
        / "zip_files"
    )
    model_downloader: S3ModelBucket = S3ModelBucket(PROD_BUCKET, test_path)

    # ACT/ASSERT
    with pytest.raises(S3RequestException):
        model_downloader.get_available_models()  # any status code != 200 should throw S3RequestException
