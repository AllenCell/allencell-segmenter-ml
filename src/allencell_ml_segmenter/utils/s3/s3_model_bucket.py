from pathlib import Path
from typing import Optional
from xml.etree import ElementTree
import requests

from allencell_ml_segmenter.utils.s3.s3_available_model import AvailableModel
from allencell_ml_segmenter.utils.s3.s3_request_exception import (
    S3RequestException,
)
from allencell_ml_segmenter.utils.s3.s3_bucket_constants import XML_NAMESPACES


class S3ModelBucket:
    def __init__(self, bucket_endpoint: str, path_to_save_models: Path):
        self._bucket_endpoint: str = bucket_endpoint
        self._path_to_save_models: Path = path_to_save_models

    def get_available_models(self) -> dict[str, AvailableModel]:
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
                self._parse_s3_xml_filelist_for_model_names(response.content)
            )
            # Create Dict of available Models
            # Where key- model_name and value- AvailableModels object (which stores the endpoint to download each model)
            available_models_dict: dict[str, AvailableModel] = {}
            for model_name in model_names:
                available_models_dict[model_name.split(".")[0]] = (
                    AvailableModel(
                        model_name,
                        self._bucket_endpoint,
                        self._path_to_save_models,
                    )
                )
            return available_models_dict
        else:
            # There was an issue with s3:ListBucket
            raise S3RequestException(
                f"S3 Download failed with status code {response.status_code}. {response.text}"
            )

    def _parse_s3_xml_filelist_for_model_names(
        self, response: bytes
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
            xml_content.find("aws_s3:Key", XML_NAMESPACES).text
            for xml_content in xml_root.findall(
                "aws_s3:Contents", XML_NAMESPACES
            )
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
