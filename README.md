# LFA-CV 🦠💊⚒
This repository contains files related with designing and training a computer vision model that would correctly classify the results of lateral flow assay (LFA) tests.

* To install relevant packages: `pip install -r requirements.txt`
* Need to add environment variable using `PYTHONPATH="path/to/your/directory:$PYTHONPATH` (or add the root directory to `sys.path` at top of each .py file)

## Project Structure
```
LFA-CV
|
|————data
|      |
|      |————self-data (collected by ourselves from 小红书)
|      |         |————image (all the images including the original and augmented ones)
|      |         |————Annotations (annotation files from label-studio)
|      |
|      |————augmented-TRAIN.csv (contain pointers to image names and coordinates of bounding boxes)
|
|————models  (.py files for implementation of models)
|————utils  (helper functions and modules)
|————trained    (store trained models)
|————config     (store model parameters using .yaml)
|————sweep      (store .yaml files for hyper-parameter tuning using Weights & Biases)
|————effdet   (efficientdet implemented by Wightman)
|————run.py   (file for training models and tuning hyper-parameters)
```
