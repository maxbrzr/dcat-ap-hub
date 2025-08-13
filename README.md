# DCAT-AP Hub

This library provides easy data handling based on metadata in DCAT-AP format.

### How To Install

```bash
pip install git+https://github.com/maxbrzr/dcat-ap-hub.git
```

### With Custom Parsing

```python
from dcat_ap_hub import download_data, apply_parsing

json_ld_metadata = "http://localhost:8081/datasets/uci-har.jsonld"
metadata = download_data(json_ld_metadata)
df = apply_parsing(metadata)
```

### With Default Pandas Parsing

```python
from dcat_ap_hub import download_data, parse_with_pandas

json_ld_metadata = "http://localhost:8081/datasets/uci-har.jsonld"
metadata = download_data(json_ld_metadata)
df = parse_with_pandas(metadata)
```