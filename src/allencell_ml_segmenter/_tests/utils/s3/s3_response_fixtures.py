import pytest


@pytest.fixture
def s3_response_listobjectv2_contents_two_models() -> str:
    """
    This function mimics the contents of a response to s3:ListObjectsv2
    if the request is made to a bucket containing two different models and a file.
    example response is from https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    and does not represent any real data.
    """
    return (
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


@pytest.fixture
def s3_response_listobjectv2_contents_duplicate_model() -> str:
    """
    This function mimics the contents of a response to s3:ListObjectsv2
    if the request is made to a bucket containing two models with duplicate names and a file.
    example response is from https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    and does not represent any real data.
    """
    return (
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
        f"    <Key>model1.zip</Key>"  # 2nd Model file name
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
