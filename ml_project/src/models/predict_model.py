# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import load

TARGET = "target"


@click.command()
@click.argument('data_path', type=click.Path(), default='data/processed/test.csv')
@click.argument('model_path', type=click.Path(), default='models/estimator.pkl')
@click.argument('result_path', type=click.Path(), default='data/processed/predict_proba.csv')
def main(data_path, model_path, result_path):
    """ Trains model, returns estimator.
    """
    logger = logging.getLogger(__name__)
    logger.info('predict with model')

    df = pd.read_csv(data_path)
    logger.info(f'file {data_path} reading successfully finished')
    estimator = load(model_path)
    logger.info(f'classifier from {model_path} successfully extracted')
    features = list(df.columns)
    if TARGET in features:
        features.remove(TARGET)
        logger.info(f'Test dataset {data_path} contains train column. Suspicious column removed')
    X = df[features] #used skiti-learn like style
    logger.info(f'Got X from dataframe according to features list')
    proba = estimator.predict_proba(X)[:, 0]
    logger.info(f'prediction successfully created')
    pd.DataFrame(proba).to_csv(result_path, header=False, index=False)
    logger.info(f'prediction successfully saved in {result_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()