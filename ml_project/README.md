ml_project
==============================

ml in prod homework 1

# ML production ready project example
## How to use
#### Dag of train pipeline is implemented with *Makefile*.
To run data preparation from console use command with args {input_filepath, output_filepath_train, output_filepath_test}:
```commandline
cd ml_project
python src/data/make_dataset.py data/raw/heart_cleveland_upload.csv data/processed/train.csv data/processed/test.csv
```
or (with Make:)
```commandline
make dataset input_filepath=data/raw/heart_cleveland_upload.csv output_filepath_train=data/processed/train.csv output_filepath_test=data/processed/test.csv
```

To run eda from console use command:
```commandline
cd ml_project
python reports/eda.py
```
or (with Make:)
```commandline
make eda
```

To train model from console use command with args {data_path, model_path}:
```commandline
cd ml_project
python src/models/train_model.py data/processed/train.csv models/estimator.pkl
```
or (with Make:)
```commandline
make train data_path=data/processed/train.csv model_path=models/estimator.pkl
```

To predict from console use command with args {data_path, model_path, result_path}:
```commandline
cd ml_project
python src/models/predict_model.py data/processed/test.csv models/estimator.pkl data/processed/predict_proba.csv
```
or (with Make:)
```commandline
make predict data_path=data/processed/test.csv model_path=models/estimator.pkl result_path=data/processed/predict_proba.csv
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
