"""Constants used when parsing and interpreting DCAT-AP metadata."""

# Type URI used to identify model entries in the source metadata.
MODEL_TYPE = "http://www.w3.org/ns/mls#Model"

# HammerHAI vocabulary, shared with the harvested catalogs on data.hammerhai.eu.
HAMMERHAI_NAMESPACE = "https://data.hammerhai.eu/vocab/"

# Profile of a processor resource: an ``rdfs:Resource`` linked from the dataset
# with ``dct:relation`` whose script turns the raw files into the AI-ready split.
PROCESSOR_PROFILE_URI = HAMMERHAI_NAMESPACE + "Processor"
# Conventional file name of a processor script.
PROCESSOR_FILENAME = "processor.py"
# Files every processor must write to its output directory.
SPLIT_FILES = ("train.csv", "test.csv")

# Namespaces used to normalize JSON-LD terms to the prefixes this parser reads.
# Keys are the canonical prefixes; values are their namespace IRIs.
CANONICAL_PREFIXES = {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "mls": "http://www.w3.org/ns/mls#",
    "hammerhai": HAMMERHAI_NAMESPACE,
}

# Model metadata profiles for backend role inference.
HF_METADATA_PROFILE_URI = "http://example.org/profiles/HuggingFaceMetadata"
ONNX_METADATA_PROFILE_URI = "http://example.org/profiles/OnnxMetadata"
SKLEARN_METADATA_PROFILE_URI = "http://example.org/profiles/SklearnMetadata"

# Legacy format markers kept for metadata compatibility with older records.
# New metadata should prefer the profile URIs above.
HF_FORMAT = "https://huggingface.co/formats/repository"
ONNX_FORMAT = "ONNX"
