# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

TARGET = "target"
RANDOM_STATE = 103
N_CROSSVAL_FOLDS = 5


@click.command()
@click.argument('data_path', type=click.Path(), default='data/processed/train.csv')
@click.argument('model_path', type=click.Path(), default='models/estimator.pkl')
def train_model(data_path: str, model_path: str):
    """ Trains model, saves estimator.
    """
    instance_logger_object = logging.getLogger(__name__)
    instance_logger_object.info('training model')

    df = pd.read_csv(data_path)
    instance_logger_object.info(f'file {data_path} reading successfully finished')
    estimator = LogisticRegression(random_state=RANDOM_STATE)
    features = list(df.columns)
    features.remove(TARGET)
    X = df[features]  # used skiti-learn like style
    y = df[TARGET]  # used skiti-learn like style
    instance_logger_object.info(f'df splitted on X, y according to features/target list')
    instance_logger_object.info(f'start cross_validation:...')
    roc_auc_crossval_scores = cross_val_score(estimator, X, y, cv=N_CROSSVAL_FOLDS, scoring='roc_auc', verbose=0)
    instance_logger_object.info(f'roc_auc cross_val results: mean:{roc_auc_crossval_scores.mean()}, '
                                f'stddev:{roc_auc_crossval_scores.std()}'
                                f' on {N_CROSSVAL_FOLDS} folds')
    estimator.fit(X, y)
    instance_logger_object.info(f'model successfully fitted')
    train_roc_auc_score = roc_auc_score(y, estimator.predict_proba(X)[:, 1])
    instance_logger_object.info(f'fitted model roc_auc score is:{train_roc_auc_score}')
    dump(estimator, model_path)
    instance_logger_object.info(f'model successfully dumped in {model_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train_model()
