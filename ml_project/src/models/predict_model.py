# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from joblib import load

TARGET = "target"


@click.command()
@click.argument('data_path', type=click.Path(), default='data/processed/test.csv')
@click.argument('model_path', type=click.Path(), default='models/estimator.pkl')
@click.argument('result_path', type=click.Path(), default='data/processed/predict_proba.csv')
def make_prediction(data_path: str, model_path: str, result_path: str):
    """ Trains model, saves estimator.
    """
    instance_logger_object = logging.getLogger(__name__)
    instance_logger_object.info('predict with model')

    dataframe_for_prediction = pd.read_csv(data_path)
    instance_logger_object.info(f'file {data_path} reading successfully finished')
    estimator = load(model_path)
    instance_logger_object.info(f'classifier from {model_path} successfully extracted')
    features = list(dataframe_for_prediction.columns)
    if TARGET in features:
        features.remove(TARGET)
        instance_logger_object.info(f'Test dataset {data_path} contains train column. Suspicious column removed')
    X = dataframe_for_prediction[features]  # used skiti-learn like style
    instance_logger_object.info(f'Got X from dataframe according to features list')
    proba = estimator.predict_proba(X)[:, 0]
    instance_logger_object.info(f'prediction successfully created')
    pd.DataFrame(proba).to_csv(result_path, header=False, index=False)
    instance_logger_object.info(f'prediction successfully saved in {result_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    make_prediction()
