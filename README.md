# DCAT-AP Hub

This library enables easy downloading and loading of datasets and models whose metadata is provided in the DCAT-AP format. Currently, only JSON-LD is supported.

### How To Install

<!-- ```bash
pip install git+https://github.com/maxbrzr/dcat-ap-hub.git
``` -->

```bash
pip install dcat-ap-hub
```

### How To Load Datasets

```python
from dcat_ap_hub import download_data, load_data

url = "https://data.europa.eu/api/hub/repo/datasets/7b715249-0c76-4592-9df6-f36b9a47f6e5.jsonld"

dataset_dir = download_data(url, base_dir="./datasets")
data = load_data(dataset_dir, summarize=True, lazy=True)
```

### How To Load Huggingface Models

```python
from dcat_ap_hub import load_hf_model

url = "https://ki-daten.hlrs.de/hub/repo/datasets/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837.jsonld"

model, processor, metadata = load_hf_model(url)
```

### Funding

This project was developed using resources from the HammerHAI project, an EU co-funded AI Factory initiative operated by the High-Performance Computing Center Stuttgart and supported by the European Commission as well as German federal and state ministries. It is funded by the European High Performance Computing Joint Undertaking under Grant Agreement No. 101234027.

<!-- ### With Custom Parsing

```python
from dcat_ap_hub import download_data, apply_parsing

json_ld_metadata = "http://localhost:8081/datasets/uci-har.jsonld"
metadata = download_data(json_ld_metadata)
df = apply_parsing(metadata)
``` -->