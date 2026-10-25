# DCAT-AP Hub

`dcat-ap-hub` is a Python library for working with datasets and pretrained models described using DCAT-AP metadata.
It is built around a practical workflow that resolves metadata, downloads artifacts, and loads datasets or models through a single interface.
Currently, metadata parsing supports **JSON-LD** from direct URLs, content negotiation, and local files.

### Typical Workflow

1. Retrieve dataset metadata in DCAT-AP from:
   - remote JSON-LD URLs (`Dataset.from_url(...)`)
   - local metadata files (`Dataset.from_file(...)`)
   - local directories that contain metadata files (`Dataset.from_directory(...)`)

2. Download files referenced by distributions and related resources (`dcat:downloadURL`) into a local dataset directory.

3. Load files or models for use in code:
   - Load files as a lazy `FileCollection` with built-in loaders for common formats such as CSV, Excel, JSON, Parquet, images, PDF, text, HTML/XML, and NumPy arrays.
   - Load pretrained models through Hugging Face, ONNX, or sklearn-style model scripts.

### Benchmarking With Catalogues

Optionally, related resources can be used to attach a processor script that is detected automatically and applied to transform raw files.
This enables the definition of multi-dataset benchmarks as DCAT-AP catalogues, since benchmarking requires each dataset to provide a fixed train-test split, which can be generated through these processor scripts.

### Requirements for Metadata

- Each dataset metadata record must include a `dcat:Dataset` entry.
- Entries with `@type` set to `mls:Model` are treated as models.
- Roles for distributions (`dcat:Distribution`) and related resources (`rdfs:Resource`) can be defined through `dct:conformsTo` and/or `dct:format`, allowing the specification of model types or processors.
- The `dcat:downloadURL` field identifies the files to be downloaded.
- Prefixes are normalized: `dcterms:`, full IRIs, or a custom `@context` work the
  same as `dct:` / `dcat:` / `rdfs:`. Both `{"@graph": [...]}` and single-node
  documents are accepted.

### Processor Resources

A processor is an `rdfs:Resource` entry. Mark it with
`dct:conformsTo: "http://example.org/profiles/Processor"` and link it from the
dataset with `dct:relation` so it is preferred when several resources exist:

```json
{
  "@graph": [
    {
      "@id": "https://example.org/dataset/1",
      "@type": "dcat:Dataset",
      "dct:title": "Example",
      "dct:relation": {"@id": "https://example.org/dataset/1#processor"}
    },
    {
      "@id": "https://example.org/dataset/1#processor",
      "@type": "rdfs:Resource",
      "dct:title": "parser.py",
      "dct:conformsTo": "http://example.org/profiles/Processor",
      "dcat:downloadURL": "https://example.org/files/parser.py"
    }
  ]
}
```

The processor script does not need to import dcat-ap-hub. It must provide one of
the following, checked in this order:

1. a class inheriting `dcat_ap_hub.DataProcessor`;
2. a class defined in the script with a `process(input_files, output_dir)` method;
3. a module-level function:

```python
def process(input_files: list[Path], output_dir: Path) -> None:
    ...  # read raw files, write e.g. train.csv and test.csv to output_dir
```

`input_files` are the top-level files of the dataset folder, excluding the
processor script, an attached notebook, and `dcat-metadata.jsonld`.

To process a local folder, place the raw files, the processor script, and a
`dcat-metadata.jsonld` describing them in one directory; the processor file may
be named after the resource title (`parser.py`):

```python
ds = Dataset.from_directory("path/to/folder")
ds.process()  # writes path/to/folder/processed/
```

### How To Install

```bash
# Base install (datasets, processing)
pip install dcat-ap-hub

# Install with ONNX model loading support
pip install "dcat-ap-hub[onnx]"

# Install with Hugging Face model loading support
pip install "dcat-ap-hub[huggingface]"
```

### Example of Loading a Dataset

```python
from dcat_ap_hub import Dataset

url = "https://ki-daten.hlrs.de/de/dataset/https-piveau-io-set-data-predictive-maintenance-ttl"

ds = Dataset.from_url(url)
files = ds.download(data_dir="./data")
```

### Example of Loading a Huggingface Model

```python
from dcat_ap_hub import Dataset

url = "https://ki-daten.hlrs.de/de/model/prajjwal1-bert-tiny"

ds = Dataset.from_url(url)
files = ds.download(data_dir="./data")
model, processor, metadata = ds.load_model(model_dir="./models")
```

### Example of Loading a SKLearn Model

```python
from dcat_ap_hub import Dataset

url = "https://ki-daten.hlrs.de/de/model/https-piveau-io-set-data-pre-trained-transformer"

ds = Dataset.from_url(url)
files = ds.download(data_dir="./data")
model = ds.load_model(model_dir="./models")
```

### Example of Processing a Dataset if Available

```python
from dcat_ap_hub import Dataset

url = "https://ki-daten.hlrs.de/de/dataset/https-piveau-io-set-data-predictive-maintenance-ttl"

ds = Dataset.from_url(url)
files = ds.download(data_dir="./data")
processed = ds.process(processed_dir="./processed")
```

### Funding

This project was developed using resources from the HammerHAI project, an EU co-funded AI Factory initiative operated by the High-Performance Computing Center Stuttgart and supported by the European Commission as well as German federal and state ministries. It is funded by the European High Performance Computing Joint Undertaking under Grant Agreement No. 101234027.
