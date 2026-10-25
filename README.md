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

A processor is an `rdfs:Resource` marked with
`dct:conformsTo: https://data.hammerhai.eu/vocab/Processor` (`hammerhai:Processor`)
and linked from the dataset with `dct:relation`, so it is preferred when several
resources exist. Only resources with this profile are run as processors.

```json
{
  "@context": {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dct": "http://purl.org/dc/terms/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
  },
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
      "dct:title": "processor.py",
      "dct:conformsTo": {"@id": "https://data.hammerhai.eu/vocab/Processor"},
      "dcat:downloadURL": "processor.py"
    }
  ]
}
```

A relative `dcat:downloadURL` names a file next to the record: in a local folder,
or, for a record fetched by URL, relative to that URL. A dataset folder can
therefore be published as is. `add_processor(record)` adds such an entry to any
record, whatever prefixes it uses.

The processor script does not need to import dcat-ap-hub. It must define a
module-level function (or, alternatively, a `dcat_ap_hub.DataProcessor`
subclass):

```python
def process(data_dir: Path, output_dir: Path) -> None:
    ...  # read the raw files in data_dir, write train.csv and test.csv
```

`data_dir` is the dataset folder. Besides the raw files, possibly in sub-folders
extracted from archives, it holds the processor, `dcat-metadata.jsonld`, and
`output_dir` (`data_dir/processed`); a processor reads only the files it needs.
It must write `train.csv` and `test.csv`; `process()` fails otherwise, and only a
complete split counts as already processed.

To process a local folder, place the raw files, `processor.py`, and a
`dcat-metadata.jsonld` declaring it in one directory:

```python
ds = Dataset.from_directory("path/to/folder")
ds.process()  # writes path/to/folder/processed/train.csv and test.csv
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

Any record with a `hammerhai:Processor` works, for example an AI-ready dataset
produced by the agentic data engineer:

```python
from dcat_ap_hub import Dataset

url = "https://example.org/ai-ready/dcat-metadata.jsonld"  # an AI-ready record

ds = Dataset.from_url(url)
files = ds.download(data_dir="./data")
processed = ds.process()  # ./data/<title>/processed/train.csv and test.csv

# A run folder of the agentic data engineer works before publishing:
local = Dataset.from_directory("output/<example>/runs/<run>/data")
local.process()
```

### Funding

This project was developed using resources from the HammerHAI project, an EU co-funded AI Factory initiative operated by the High-Performance Computing Center Stuttgart and supported by the European Commission as well as German federal and state ministries. It is funded by the European High Performance Computing Joint Undertaking under Grant Agreement No. 101234027.
