# heritage-connector-nlp

Text processing for the [Heritage Connector](https://github.com/TheScienceMuseum/heritage-connector): a set of NLP utilities for the Heritage sector.

**--- IN DEVELOPMENT ---**

Includes:

- information extraction (NER, NEL, relation classification)
- labelling ([Label Studio](https://labelstud.io/))
- test suite for models


## Usage

### Label Studio

**Setting up (first time):**

1. Run `label-studio start labelling --init`, which will start up Label Studio and take you to a configuration wizard. 
2. Select *Named Entity Recognition* from the top menu, and fill in the entity types you want to annotate

**Running:** Run `label-studio start labelling` from the root directory.

**Useful parameters:**

- `--sampling=uniform`: have Label Studio show documents in a random order
- `--label-config label_studio_config_sample.xml`: load config from a file
