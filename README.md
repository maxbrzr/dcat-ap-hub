# DCAT-AP Hub

This library enables easy downloading and loading of datasets and models whose metadata is provided in the DCAT-AP format. Currently, only JSON-LD is supported.

### How To Install

```bash
# Base install (datasets, processing)
pip install dcat-ap-hub

# Install with ONNX model loading support
pip install "dcat-ap-hub[onnx]"

# Install with Hugging Face model loading support
pip install "dcat-ap-hub[huggingface]"
```

### How To Load Datasets

```python
from dcat_ap_hub import Dataset

url = "https://data.europa.eu/api/hub/repo/datasets/7b715249-0c76-4592-9df6-f36b9a47f6e5.jsonld"

ds = Dataset.from_url(url)
files = ds.download(data_dir="./data")
```

### How To Load SKLearn Models

```python
from dcat_ap_hub import Dataset

url = "https://ki-daten.hlrs.de/hub/repo/datasets/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837.jsonld"

ds = Dataset.from_url(url)
ds.download(data_dir="./data")
model = ds.load_model(model_dir="./models")
```

### How To Load Huggingface Models

```python
from dcat_ap_hub import Dataset

url = "https://ki-daten.hlrs.de/hub/repo/datasets/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837.jsonld"

ds = Dataset.from_url(url)
ds.download(data_dir="./data")
model, processor, metadata = ds.load_model(model_dir="./models")
```

### How To Process Datasets If Supported

```python
from dcat_ap_hub import Dataset

url = "https://data.europa.eu/api/hub/repo/datasets/7b715249-0c76-4592-9df6-f36b9a47f6e5.jsonld"

ds = Dataset.from_url(url)
ds.download(data_dir="./data")
files = ds.process(processed_dir="./processed")
```

### Funding

This project was developed using resources from the HammerHAI project, an EU co-funded AI Factory initiative operated by the High-Performance Computing Center Stuttgart and supported by the European Commission as well as German federal and state ministries. It is funded by the European High Performance Computing Joint Undertaking under Grant Agreement No. 101234027.

