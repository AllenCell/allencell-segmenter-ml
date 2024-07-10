import pytest
import responses

from allencell_ml_segmenter.utils.s3 import S3ModelDownloader, AvailableModels
from allencell_ml_segmenter.utils.s3.s3_model_downloader import (
    STG_BUCKET,
    PROD_BUCKET,
)
from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)


def test_init_s3_downloader_with_staging() -> None:
    # Act
    downloader: S3ModelDownloader = S3ModelDownloader(staging=True)

    # Assert
    assert downloader.get_bucket_endpoint() == STG_BUCKET


def test_init_s3_downloader_with_prod() -> None:
    # Act
    # No params defaults to prod bucket
    downloader: S3ModelDownloader = S3ModelDownloader()

    # Assert
    assert downloader.get_bucket_endpoint() == PROD_BUCKET


@responses.activate
def test_get_available_models() -> None:
    # ARRANGE
    test_url: str = "http://tests3bucketendpoint.com"
    # The following line mimics what s3 would send back from the ListObjectV2 http request
    # example response is from https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    # and does not represent any real data.
    xml_response: str = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
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
        f"    <Key>model2.zip</Key>"  # 2nd Model file name
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
            "url": f"{test_url}?list-type=2",
            "body": xml_response,
            "status": 200,
            "content_type": "application/xml",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )

    model_downloader: S3ModelDownloader = S3ModelDownloader(test_url=test_url)

    # ACT
    available_models_dict: dict[str, AvailableModels] = (
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
        == f"{test_url}/model1.zip"
    )
    assert available_models_dict["model2"].get_name() == "model2.zip"
    assert (
        available_models_dict["model2"].get_object_url()
        == f"{test_url}/model2.zip"
    )


@responses.activate
def test_get_available_models_duplicate_file_error() -> None:
    # ARRANGE
    test_url: str = "http://testbucketendpoint.com"
    # The following line mimics what s3 would send back from the ListObjectV2 http request
    # example response is from https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    # and does not represent any real data.
    xml_response: str = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
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
        f"    <Key>model1.zip</Key>"  # 1st Model file name (duplicated)
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
            "url": f"{test_url}?list-type=2",
            "body": xml_response,
            "status": 200,
            "content_type": "application/xml",
            "adding_headers": {"X-Foo": "Bar"},
        }
    )

    model_downloader: S3ModelDownloader = S3ModelDownloader(test_url=test_url)

    # ACT/ASSERT
    with pytest.raises(ValueError):
        # if there are models with duplicate names on s3 somehow- we show an error
        model_downloader.get_available_models()


@responses.activate
def test_get_available_models_bad_request() -> None:
    # ARRANGE
    test_url: str = "http://testbucketendpoint.com"
    error_response: str = "error"
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

    model_downloader: S3ModelDownloader = S3ModelDownloader(test_url=test_url)

    # ACT/ASSERT
    with pytest.raises(S3RequestException):
        model_downloader.get_available_models()  # any status code != 200 should throw S3RequestException
