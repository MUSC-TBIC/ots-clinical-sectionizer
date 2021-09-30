medspaCy
========

The clinical note sectionizer provided by medspaCy is rule-based so no
training or model are required.

The `types/` directory in this repository contains CAS XMI type
definitions used during processing.

```
python3 -m pip install -r medspacy/requirements.txt

python3 medspacy/medspaCy_sectionizer.py \
  --types-dir <directory to CAS XMI type definitions> \
  --input-dir <plain text input directory> \
  --output-dir <output directory>
```

SVM spaCy
=========

Train a Model
-------------

The SVM trainer splits the training corpus into a further 70/30
(train/eval) set to allow pre-test verification of performance of the
parameter grid search used with the rbf kernel.

A csv file (`cleaned_processed_dataset.csv`) is written to the output
directory for manual inspection.

```
python3 -m pip install -r svm-spacy/requirements.txt

## Example of a training run
python3 svm-spacy/spaCY-SVM_sectionizer.py \
    --input-dir <brat annotated corpus> \
    --output-dir <XMI formatted output directory>" \
    --model-dir <output directory for pkl model file> \
    --train \
    --svm-kernel [ "linear" | "rbf" ]
```

Annotate a Corpus
-----------------

The `types/` directory in this repository contains CAS XMI type
definitions used during processing.

If the training flag is not set, the default action is to annotate the
corpus.

```
## Example of an annotation run
python3 svm-spacy/spaCY-SVM_sectionizer.py \
    --types-dir <directory to CAS XMI type definitions> \
    --input-dir <plain text corpus> \
    --output-dir <XMI formatted output directory>" \
    --model-dir <input directory containing pkl model file> \
    --svm-kernel [ "linear" | "rbf" ]

```
