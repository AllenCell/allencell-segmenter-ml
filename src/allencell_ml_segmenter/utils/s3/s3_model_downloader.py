from allencell_ml_segmenter.utils.s3 import AvailableModels
from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)
from typing import Optional
from xml.etree import ElementTree
import requests

# CONSTANTS RELATED TO MODEL DOWNLOADS
# Enable Model Downloads on plugin
ENABLE_MODEL_DOWNLOADS = True
# Endpoint for prod bucket
BUCKET_ENDPOINT = (
    "https://production-aics-ml-segmenter-models.s3.us-west-2.amazonaws.com"
)
# Endpoint for stg bucket
STG_BUCKET_ENDPOINT = ""


class S3ModelDownloader:
    def __init__(self, staging=False, test_url: Optional[str] = None):
        self._bucket_endpoint: str
        # if a test_url is provided set that as the bucket endpoint
        if test_url:
            self._bucket_endpoint = test_url
        else:
            self._bucket_endpoint = (
                STG_BUCKET_ENDPOINT if staging else BUCKET_ENDPOINT
            )

    def get_available_models(self) -> dict[str, AvailableModels]:
        """
        Get a dict of the filenames of available models on s3.

        Uses a http request to get all available models on s3. This function assumes all models will be in .zip files
        (and ignores all files that are not zip files). Any errors with the request throws a S3RequestException.

        Returns dictionary containing model name keys, and AvailableModels values.
        """
        # request parameter to list all objects in bucket
        req_params: str = "list-type=2"

        # request all bucket objects
        response: requests.Response = requests.get(
            f"{self._bucket_endpoint}?{req_params}"
        )

        if response.status_code == 200:
            # parse XML response with key
            model_names: set[str] = (
                self._parse_s3_xml_filelist_for_model_names(
                    response.content, ".//Contents"
                )
            )
            # Create Dict of available Models
            # Where key- model_name and value- AvailableModels object (which stores the endpoint to download each model)
            available_models_dict: dict[str, AvailableModels] = {}
            for model_name in model_names:
                available_models_dict[model_name.split(".")[0]] = (
                    AvailableModels(model_name, self._bucket_endpoint)
                )
            return available_models_dict
        else:
            # There was an issue with s3:ListBucket
            raise S3RequestException(
                f"S3 Download failed with status code {response.status_code}. {response.text}"
            )

    def _parse_s3_xml_filelist_for_model_names(
        self, response: bytes, element_name: str
    ) -> set[str]:
        """
        Parse XML response from s3 containing list of available objects for model names.
        Returns list of available models (which are .zip files) from s3's XML response.
        """
        # parse XML response
        xml_root: ElementTree = ElementTree.fromstring(response)
        available: set[str] = set()
        # find all XML elements which match :param element_name: and get its "Key" element
        # which contains the object's filename
        all_s3_objects: list[str] = [
            xml_content.find("Key").text
            for xml_content in xml_root.findall(element_name)
        ]
        for file in all_s3_objects:
            # append file name if it is a zip file- these are models stored on s3
            if file.endswith(".zip"):
                # S3 allows duplicate files- ensure we have unique names for all models on s3.
                if file in available:
                    raise ValueError(
                        f"S3 bucket {self._bucket_endpoint} contains duplicate models for name: {file}"
                    )
                available.add(file)
        return available

    def get_bucket_endpoint(self) -> str:
        return self._bucket_endpoint
