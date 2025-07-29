# Description

Human Activity Recognition database built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors.

### How To Install

```bash
pip install git+https://github.com/maxbrzr/dcat-ap-hub.git
pip install git+https://github.com/username/parser_repo.git
```

### How To Use

```python
from dcat_ap_hub import download_data, apply_parsing

json_ld_metadata = "http://localhost:8081/datasets/uci-har
metadata = download_data(json_ld_metadata)
df = apply_parsing(metadata)
```