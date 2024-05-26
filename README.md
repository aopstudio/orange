# Orange
## Install environment
Use the command:
```sh
pip install -r requirements.txt
```
## Folders and files
`instance_output/train.txt` is the pre-training corpus for entity generation(because of the file size limit in github, please download this file from https://drive.google.com/file/d/1tC4mNlDxzywnTdrqfRSvXDCZiTG9kVLb/view?usp=sharing)
`relation_output/train.txt` is the pre-training corpus for relation generation
`output/validation.txt` is the OpenRelation600 dataset

## Pre-train
Use the command
```python
python train_entity.py
```
to pre-train the entity generation model.
The pre-trained model will be saved in`entity_generator/` folder.

Use the command
```python
python train_relation.py
```
to pre-train the relation generation model.
The pre-trained model will be saved in`relation_generator/` folder

## Evaluate
Make sure the model file path is right in `inductor.py` and run the command
```sh
python evaluation.py --task openrelation600
```
to evaluate the model.

## Model download
Visit https://drive.google.com/file/d/1LzND-P2LW4Bwlt1OQYGJ9MMqoiVK_iLR/view?usp=sharing to download the relation generation model
Visit https://drive.google.com/file/d/1ow5gF-nLHthDKipXblLukChi_syFliS0/view?usp=sharing to download the entity generation model