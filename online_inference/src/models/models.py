import logging
from typing import List, Union

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, conlist
from sklearn.linear_model import LogisticRegression

instance_logger_object = logging.getLogger("models.predict")


class ModelInput(BaseModel):
    """Model input dataclass"""
    data: List[conlist(Union[float, int, None], min_items=13, max_items=13)]
    feature_names: List[str]


class ModelOutput(BaseModel):
    """Model output dataclass"""
    condition: List[float]


def make_prediction(
    data: List,
    feature_names: List[str],
    model: LogisticRegression
) -> ModelOutput:
    """Predict function"""
    instance_logger_object.info('Checking income data...')
    if len(data) == 0:
        raise HTTPException(
            status_code=400, detail="Received empty input"
        )
    instance_logger_object.info('Data is correct')
    data = pd.DataFrame(np.array(data), columns=feature_names)
    instance_logger_object.info('Making prediction...')
    prediction = list(model.predict(data))
    instance_logger_object.info('Prediction successfully made')
    return ModelOutput(condition=prediction)
