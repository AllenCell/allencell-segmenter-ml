# CONSTANTS RELATED TO MODEL DOWNLOADS
# Enable Model Downloads on plugin
ENABLE_MODEL_DOWNLOADS = True  # TODO: ENABLE WHEN MODELS UPLOADED
# Endpoint for prod bucket
PROD_BUCKET = (
    "https://production-aics-ml-segmenter-models.s3.us-west-2.amazonaws.com"
)
# Endpoint for stg bucket
STG_BUCKET = (
    "https://staging-aics-ml-segmenter-models.s3.us-west-2.amazonaws.com"
)
# XML namespaces we might expect- currently only aws_s3
# we need this to parse XML with namespaces
# for more info: https://docs.python.org/3/library/xml.etree.elementtree.html#parsing-xml-with-namespaces
XML_NAMESPACES = {"aws_s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
