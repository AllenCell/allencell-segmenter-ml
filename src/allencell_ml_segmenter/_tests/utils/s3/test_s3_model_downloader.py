import pytest
import responses

from allencell_ml_segmenter.utils.s3 import S3ModelDownloader, AvailableModels
from allencell_ml_segmenter.utils.s3.s3_model_downloader import (
    STG_BUCKET_ENDPOINT,
    BUCKET_ENDPOINT,
)
from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)


def test_init_s3_downloader_with_staging():
    # Act
    downloader: S3ModelDownloader = S3ModelDownloader(staging=True)

    # Assert
    assert downloader.get_bucket_endpoint() == STG_BUCKET_ENDPOINT


def test_init_s3_downloader_with_prod():
    # Act
    # No params defaults to prod bucket
    downloader: S3ModelDownloader = S3ModelDownloader()

    # Assert
    assert downloader.get_bucket_endpoint() == BUCKET_ENDPOINT


@responses.activate
def test_get_available_models():
    # ARRANGE
    test_url: str = "http://testbucketendpoint.com"
    model_file_names: list[str] = ["model1.zip", "model2.zip", "model3.zip"]
    # The following line mimics what s3 would send back from the ListObjectV2 http request
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
        f"    <Key>model1.zip</Key>"
        f"    <LastModified>2014-11-21T19:40:05.000Z</LastModified>"
        f'    <ETag>"70ee1738b6b21e2c8a43f3a5ab0eee71"</ETag>'
        f"    <Size>1111</Size>"
        f"    <StorageClass>STANDARD</StorageClass>"
        f"  </Contents>"
        f"  <Contents>"
        f"    <Key>model2.zip</Key>"
        f"    <LastModified>2014-11-21T19:40:05.000Z</LastModified>"
        f'     <ETag>"70ee1738b6b21e2c8a43f3a5ab0eee71"</ETag>'
        f"    <Size>1111</Size>"
        f"    <StorageClass>STANDARD</StorageClass>"
        f"  </Contents>"
        f"  <Contents>"
        f"    <Key>model3.zip</Key>"
        f"    <LastModified>2014-11-21T19:40:05.000Z</LastModified>"
        f'    <ETag>"70ee1738b6b21e2c8a43f3a5ab0eee71"</ETag>'
        f"    <Size>1111</Size>"
        f"    <StorageClass>STANDARD</StorageClass>"
        f"  </Contents>"
        f"</ListBucketResult>"
    )
    model_downloader = S3ModelDownloader(test_url=test_url)
    # add fake xml response that is returned when we make a request to the test_url with the list-type=2 param
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{test_url}?list-type=2",
            "body": xml_response,
            "status": 200,
            "content_type": "application/xml",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )

    # ACT
    available_models_dict: dict[str, AvailableModels] = (
        model_downloader.get_available_models()
    )

    # ASSERT
    # we store only file names as the dictionary keys (without .zip)
    assert list(available_models_dict.keys()) == ["model1", "model2", "model3"]
    # dictionary values contain available models object w/ name and object url on s3
    assert available_models_dict["model1"].get_name() == "model1.zip"
    assert (
        available_models_dict["model1"].get_object_url()
        == f"{test_url}/model1.zip"
    )
    assert available_models_dict["model3"].get_name() == "model3.zip"
    assert (
        available_models_dict["model3"].get_object_url()
        == f"{test_url}/model3.zip"
    )


@responses.activate
def test_get_available_models_bad_request():
    # ARRANGE
    test_url: str = "http://testbucketendpoint.com"
    model_file_names: list[str] = ["model1.zip", "model2.zip", "model3.zip"]
    # This mimics what s3 would send back from a ListBucket http request
    error_response: str = "error"
    model_downloader = S3ModelDownloader(test_url=test_url)
    responses.add(
        **{
            "method": responses.GET,
            "url": f"{test_url}?list-type=2",
            "body": error_response,
            "status": 400,
            "content_type": "error",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )

    # ACT/ASSERT
    with pytest.raises(S3RequestException):
        available_models_dict: dict[str, AvailableModels] = (
            model_downloader.get_available_models()
        )
