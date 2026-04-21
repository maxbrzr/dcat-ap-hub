"""Constants used when parsing and interpreting DCAT-AP metadata."""

# Type URI used to identify model entries in the source metadata.
MODEL_TYPE = "http://www.w3.org/ns/mls#Model"

# Profile URI used to identify processor resources.
PROCESSOR_PROFILE_URI = "http://example.org/profiles/Processor"

# Model metadata profiles for backend role inference.
HF_METADATA_PROFILE_URI = "http://example.org/profiles/HuggingFaceMetadata"
ONNX_METADATA_PROFILE_URI = "http://example.org/profiles/OnnxMetadata"
SKLEARN_METADATA_PROFILE_URI = "http://example.org/profiles/SklearnMetadata"

# Legacy format markers kept for metadata compatibility with older records.
# New metadata should prefer the profile URIs above.
HF_FORMAT = "https://huggingface.co/formats/repository"
ONNX_FORMAT = "ONNX"
