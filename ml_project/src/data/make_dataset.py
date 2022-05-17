# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/raw/heart_cleveland_upload.csv')
@click.argument('output_filepath_train', type=click.Path(), default='data/processed/train.csv')
@click.argument('output_filepath_test', type=click.Path(), default='data/processed/test.csv')
def main(input_filepath, output_filepath_train, output_filepath_test=''):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    make_split = True

    if output_filepath_test == '':
        logger.info('test_filepath is not set: train_test_split will be skipped, data duplicated')
        output_filepath_test = output_filepath_train
        make_split = False
    else:
        logger.info(f'test_filepath is set to: {output_filepath_test}')

    df = pd.read_csv(input_filepath)
    logger.info(f'file {input_filepath} reading successfully finished')
    df.dropna(inplace=True)
    logger.info(f'nans cleaned')
    df['target'] = df['condition']
    df.drop(columns=['condition'], inplace=True)
    logger.info(f'target column renamed to "target"')
    logger.info(f'preprocessing successfully completed')

    if make_split:
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=103)
        test_df.drop(columns=['target'], inplace=True) # no labels for prediction
        logger.info(f'data splited on train/test')
    else:
        train_df = df

    train_df.to_csv(path_or_buf=output_filepath_train, index=False)
    logger.info(f'train dataset successfully dumped in {output_filepath_train}')
    if make_split:
        test_df.to_csv(path_or_buf=output_filepath_test, index=False)
        logger.info(f'test dataset successfully dumped in {output_filepath_test}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
