# Idea

- A Benchmark corresponds to a DCAT-AP catalogue
- each catlogue contains multiple DCATP-AP datasets
    - landingPage should be link to releease of dataset and explanation
    - description is ai summary of landingpage?
- each dataset contains multiple DCTAP-AP ddisibutions for different files
    - donwloadURL should be link to downloaded file
- each dataset has a DCAP-AP relation to a parser repo for this dataset, and a notebook repo / link to server with notebook for this dataset
    - notebook should use dcat-ap hub to load and parser to parse


# More

- bencmark dont only have dataset, but also models
- so running benhcmakr correponds to evaluation each model on each dataset (cross product)
- benchmarking in single notebook -> need notebook entry for a catalogue
- models pretrained? Then would need some sort of foundation model, since otherwise cross dataset generalization not possible
    - would have to be for particular modality
- models not pretrained? then benchmarking involves training
- but seems like manufacturing isnt a domain for foundation models yet. so model should maybe only refer to architecture?

# Example

- one benmark for a few manufacturing datasets
- one benchmark for a few har dataset
- can build parsers for each, but in github repo (can be single since download url -> dcat-ap parsers?)
    - must be identified with some id
- can build noebook for each, but for now only but in same repo as parsers, since jupyter server does not exist yet


# Crwaling

- update metadata from crawling since bad quality

# UI integration

- space for parse, if not available then say so
- space for notebook, if not available then say so



# Format for each Modality

- for benchmarking, we need standardized format for different modalities
- look how manufacturing dtaa looks at awesome datasets https://github.com/jonathanwvd/awesome-industrial-datasets
- loook for formats on huggingface https://huggingface.co/docs/datasets/repository_structure


# Deprecated

- using catalogue record for parse / notebook
    - used for describing metadata of dataset
    - no good attributes for links
    - one-to-one mapping with Datasets -> so cant make 2, one foreach ( parser and notebook)

# Convert mlcroissant to DCAT-AP

- good for kaggle


# Tasks

1. choose benchmark -> manufacturing X
2. choose datasets X
3. write parse and notebook -> make repo X
4. write metadata for benchmark and upload to test instance X
5. fix library code to work with related resources x
5. integrate into frontend 
6. benchmark notebook
7. update names and descriptions