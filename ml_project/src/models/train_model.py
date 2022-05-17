# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

TARGET = "target"


@click.command()
@click.argument('data_path', type=click.Path(), default='data/processed/train.csv')
@click.argument('model_path', type=click.Path(), default='models/estimator.pkl')
def main(data_path, model_path):
    """ Trains model, returns estimator.
    """
    logger = logging.getLogger(__name__)
    logger.info('training model')

    df = pd.read_csv(data_path)
    logger.info(f'file {data_path} reading successfully finished')
    estimator = LogisticRegression(random_state=103)
    features = list(df.columns)
    features.remove(TARGET)
    X = df[features] #used skiti-learn like style
    y = df[TARGET] #used skiti-learn like style
    logger.info(f'df splitted on X, y according to features/target list')
    estimator.fit(X, y)
    logger.info(f'model successfully fitted')
    dump(estimator, model_path)
    logger.info(f'model successfully dumped in {model_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
