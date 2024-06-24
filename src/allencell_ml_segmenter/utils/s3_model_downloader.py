from pathlib import Path
from typing import Optional

import boto3

from allencell_ml_segmenter.main.experiments_model import ExperimentsModel

# need this to access staging bucket, but wont need for public bucket
BUCKET_NAME = 'staging-aics-ml-segmenter-models'
PROFILE_NAME = "filestorage_production_admin"


class S3ModelDownloader:
    def __init__(self, staging=False):
        self._bucket = self._get_staging_bucket_object() if staging else self._get_bucket_object()
        self._available_models: Optional[set[str]] = None # Available after call ti get_available_models()


    def _get_staging_bucket_object(self):
        session = boto3.Session(profile_name=PROFILE_NAME)
        s3 = session.resource("s3")
        return s3.Bucket(BUCKET_NAME)


    def _get_bucket_object(self):
        # Implement this w/ public bucket when available
        pass

    def get_available_models(self) -> set[str]:
        unique_toplevel_dir = set()
        for bucket_object in self._bucket.objects.all():
            unique_toplevel_dir.add(bucket_object.key.split("/")[0])

        self._available_models = unique_toplevel_dir
        return unique_toplevel_dir

    def download_model_to(self, model: str, path: Path) -> None:
        if self._available_models is None:
            raise ValueError("Call get_available_models() first to check which models are available")
        if model not in self._available_models:
            raise ValueError("model name provided is not available on s3 bucket")\


        # S3 will download all of these into the top level directory so we need to create the folders ourselves
        # and download this way
        bucket_obj_for_download = list(self._bucket.objects.filter(Prefix = model))
        for obj in bucket_obj_for_download:
            # remove file name from object key
            obj_path = Path(obj.key).parents[0]

            # replicate directory structure from s3 locally
            (path / obj_path).mkdir(parents=True, exist_ok=True)

            # save file with full path
            self._bucket.download_file(obj.key, path / obj.key)





