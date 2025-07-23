# DCAT-AP Hub

This library provides easy data handling based on metadata in DCAT-AP format.

### How To Use

```python
from dcat_ap_hub import download

path, parse = download(
    dataset_metadata_handle="http://dataset_metadata.jsonld"
    parser_metadata_handle="http://parser_metadata.jsonld"
    base_dir="./datasets"
)
```