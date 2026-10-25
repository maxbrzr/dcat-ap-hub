"""Constants used when parsing and interpreting DCAT-AP metadata."""

# Type URI used to identify model entries in the source metadata.
MODEL_TYPE = "http://www.w3.org/ns/mls#Model"

# Profile URI used to identify processor resources.
PROCESSOR_PROFILE_URI = "http://example.org/profiles/Processor"
# All profile URIs accepted as processor markers. Add a stable, resolvable URI
# here (and switch PROCESSOR_PROFILE_URI to it) once one is registered; keep the
# legacy value so existing metadata stays valid.
PROCESSOR_PROFILE_URIS = frozenset({PROCESSOR_PROFILE_URI})

# Namespaces used to normalize JSON-LD terms to the prefixes this parser reads.
# Keys are the canonical prefixes; values are their namespace IRIs.
CANONICAL_PREFIXES = {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "mls": "http://www.w3.org/ns/mls#",
}

# Model metadata profiles for backend role inference.
HF_METADATA_PROFILE_URI = "http://example.org/profiles/HuggingFaceMetadata"
ONNX_METADATA_PROFILE_URI = "http://example.org/profiles/OnnxMetadata"
SKLEARN_METADATA_PROFILE_URI = "http://example.org/profiles/SklearnMetadata"

# Legacy format markers kept for metadata compatibility with older records.
# New metadata should prefer the profile URIs above.
HF_FORMAT = "https://huggingface.co/formats/repository"
ONNX_FORMAT = "ONNX"
