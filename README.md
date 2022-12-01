# Hathi Scripts
This repository contains scripts useful for manipulating and analyzing large collections of HathiTrust book data with minimal user control. These scripts ease the pain of working with Data Capsules.

## Scripts Included
1. `get-files.py` Given a list of HTIDs, randomly sample and save pages for each HTID. 
2. `get_marc_metadata.py` Given a list of HTIDs, download and save MARC metadata for each HTID.
3. `finetune.py` Given a directory of randomly sampled pages, fine-tune a HuggingFace classifier.
4. `classify.py` Given a HuggingFace classifier and a list of HTIDs, classify volumes with said classifier.
