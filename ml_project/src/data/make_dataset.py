# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 103


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/raw/heart_cleveland_upload.csv')
@click.argument('output_filepath_train', type=click.Path(), default='data/processed/train.csv')
@click.argument('output_filepath_test', type=click.Path(), default='data/processed/test.csv')
def dataset_preparation(input_filepath: str, output_filepath_train: str, output_filepath_test: str = ''):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    instance_logger_object = logging.getLogger(__name__)
    instance_logger_object.info('making final data set from raw data')
    make_split = True

    if output_filepath_test == '':
        instance_logger_object.info('test_filepath is not set: train_test_split will be skipped, data duplicated')
        output_filepath_test = output_filepath_train
        make_split = False
    else:
        instance_logger_object.info(f'test_filepath is set to: {output_filepath_test}')

    raw_dataframe = pd.read_csv(input_filepath)
    instance_logger_object.info(f'file {input_filepath} reading successfully finished')
    raw_dataframe.dropna(inplace=True)
    instance_logger_object.info(f'nans cleaned')
    raw_dataframe['target'] = raw_dataframe['condition']
    raw_dataframe.drop(columns=['condition'], inplace=True)
    instance_logger_object.info(f'target column renamed to "target"')
    instance_logger_object.info(f'preprocessing successfully completed')

    if make_split:
        train_df, test_df = train_test_split(raw_dataframe, test_size=0.2, stratify=raw_dataframe['target'],
                                             random_state=RANDOM_STATE)
        test_df.drop(columns=['target'], inplace=True)  # no labels for prediction
        instance_logger_object.info(f'data splited on train/test')
    else:
        train_df = raw_dataframe

    train_df.to_csv(path_or_buf=output_filepath_train, index=False)
    instance_logger_object.info(f'train dataset successfully dumped in {output_filepath_train}')
    if make_split:
        test_df.to_csv(path_or_buf=output_filepath_test, index=False)
        instance_logger_object.info(f'test dataset successfully dumped in {output_filepath_test}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    dataset_preparation()
