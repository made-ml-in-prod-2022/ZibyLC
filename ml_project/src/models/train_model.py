import click
import pandas as pd
import logging
import pickle
from sklearn.model_selection import train_test_split

@click.command()
@click.argument('model')
@click.argument('data_path')
@click.argument('features')
@click.argument('target')
@click.argument('params')
def train(model, data_path, features, target, params):
    logger = logging.getLogger('Train')
    df = pd.read_csv(data_path)
    logger.info('read dataset at {}'.format(data_path))
    x_train, x_valid, y_train, y_valid = train_test_split(df[features], target,
                                                            train_size=params.train_size,
                                                            random_state=params.random_state,
                                                            shuffle=params.shuffle, stratify=target)

    logger.info('Split dataset\n\t{}'.format(split_params))
    model = model(x_train, y_train)

    logger.info('features\n\t{}'.format(params.features))
    logger.info('Preprocessing\n\t{}'.format(params.transformers))

    pred_train = predict_model(model, x_train, params)
    pred_valid = predict_model(model, x_valid, params)
    train_metrics = calculate_metrics(y_train, pred_train, params)
    valid_metrics = calculate_metrics(y_valid, pred_valid, params)
    logger.info('train metrics: \n\t{}'.format(train_metrics))
    logger.info('valid_metrics \n\t{}'.format(valid_metrics))

    with open(params.model_path, 'wb') as file:
        pickle.dump(model, file)
        logger.info('save model at {}'.format(params.model_path))


if __name__ == '__main__':
    train()

