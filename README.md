# DCAT-AP Hub

This library provides easy data handling based on metadata in DCAT-AP format.

### How To Install

```bash
pip install git+https://github.com/maxbrzr/dcat-ap-hub.git
```

### How To Use

```python
from dcat_ap_hub import download_data, apply_parsing

json_ld_metadata = "http://localhost:8081/datasets/uci-har.jsonld"
metadata = download_data(json_ld_metadata)
df = apply_parsing(metadata)
```

### TODOs

- add default parsers (e.g. using pandas)