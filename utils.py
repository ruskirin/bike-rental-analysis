import numpy as np
import pandas as pd
import time
import gc
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


SEED = 1000
TARGET = 'cnt'
MODELS = {'lin_reg': 'LinearRegression',
          'dt': 'DecisionTreeRegressor',
          'rf': 'RandomForestRegressor'}


def kfold_cv_progressive(
        data: pd.DataFrame,
        id_col: str,
        model_name,
        features: list,
        target: str = TARGET,
        k: int = 10) -> (pd.DataFrame, Tuple[str, int]):
    """
    Do multiple iterations of k-fold cross validation using @model_name while
      progressively incrementing the amount of features used from @features.
      Measure MAE, RMSE, and R2 error metrics and the time (nanoseconds) to
      fit and predict using the model. Point is to measure the decrease in
      error and time complexity of each newly added feature. Also to see if
      the order/permutation of features matters for prediction.

    NOTE: might be completely unnecessary for linear regression, because it's
      linear, but what about decision trees and random forests?

    :param data:
    :param id_col:
    :param model_name:
    :param features:
    :param target:
    :param k:
    :return: dataframe with predictions,
             tuple(last added feature, time spent fitting + predicting)
    """

    preds = data.loc[:, [id_col, target]]
    order_time = dict()

    for i in range(len(features)):
        col_name = f'{i+1}'
        pred, t = kfold_cv(
            data, id_col, model_name, features[:i+1], target, k, col_name
        )
        preds = preds.merge(pred, how='left', on=id_col)
        order_time[col_name] = features[i], t

    return preds, order_time


def kfold_cv(data: pd.DataFrame,
             id_col: str,
             model_name: str,
             features: list,
             target: str = TARGET,
             k: int = 10,
             pred_col_name: str = 'predict') -> (pd.DataFrame, int):
    """
    Manual k-fold cross validation. Keeping track of total time spent fitting
      and predicting using the model.

    :param data: dataframe with the data to train/test on
    :param id_col: name of id column for each row
    :param model_name: name of scikit-learn model to train and test
    :param features: features of @data to train the @model on
    :param target: target column to test on
    :param k: amount of randomly permuted, roughly-equal folds to create
    :param pred_col_name: name of the column of predictions
    :return: DataFrame with 2 columns: @id_col, @pred_col_name | int time in
      nanoseconds spent fitting and training @model_name
    """
    if model_name not in MODELS.keys():
        raise ValueError(f'{model_name} is not one of: {MODELS}')

    # search for any columns that were expanded into dummy variables
    expand_features = get_dummy_features(data, features)
    folds = permute_equal_kfolds(data.shape[0], k)

    preds = dict() # dictionary to store the predictions
    model_pred_time = 0 # nanoseconds spent fitting and predicting

    # Manual 10-fold cross validation
    for i, test_idx in enumerate(folds):
        # Get the k-1 training folds
        train_idx = list(set(data.index).difference(test_idx))
        train = data.iloc[train_idx, :]
        test_idx = list(test_idx)
        test = data.iloc[test_idx, :]

        # instantiate the model using its passed @model_name
        model = globals()[MODELS[model_name]]()

        gc.disable()  # turn off garbage collector for timing of predictions
        start = time.perf_counter_ns()  # time the model fitting and prediction

        model.fit(train[expand_features], train[target])
        pred = model.predict(test[expand_features])

        end = time.perf_counter_ns()
        gc.enable()
        model_pred_time += end-start

        pred = dict(zip(test[id_col].to_numpy(), pred))
        preds.update(pred)

    preds = pd.DataFrame(data=preds.items(), columns=[id_col, pred_col_name])
    return preds, model_pred_time


def get_predict_metrics(preds: pd.DataFrame,
                        target: str,
                        order_time: dict):
    """
    Calculate the metrics associated with the predictions

    :param preds: dataframe with id and @target columns, as well as prediction
      columns based on features used
    :param target:
    :param order_time:

    """
    # get all numbered columns (indicating features used for fitting/predicting
    pred_cols = preds.filter(regex=r'^\d').columns.to_numpy()
    true_out = preds.loc[:, target]

    metrics = pd.DataFrame(columns=['features', 'mae', 'rmse', 'r2', 'time_ns'])

    for col in pred_cols:
        m = {
            'features': col,
            'mae': mean_absolute_error(
                true_out, preds.loc[:, col]
            ),
            'rmse': mean_squared_error(
                true_out, preds.loc[:, col], squared=False
            ),
            'r2': r2_score(true_out, preds.loc[:, col]),
            'time_ns': order_time[col][1]
        }
        metrics.loc[metrics.shape[0]] = m

    return metrics


def get_dummy_features(data, features):
    """Check @data for dummy @features columns"""
    full = []

    for f in features:
        cols = list(data.filter(regex=fr'^{f}', axis=1).columns)
        full.extend(cols)

    return full


def permute_equal_kfolds(size, k=10):
    rng = np.random.default_rng(SEED)
    return [list(f) for f in np.array_split(rng.permutation(size), k)]


if __name__=='__main__':
    from pathlib import Path

    path = Path('data/hour.csv')
    data = pd.read_csv(path)
    features = ['temp', 'season', 'windspeed', 'workingday', 'weekday', 'holiday',
       'weathersit', 'hum']
    target = 'cnt'

    cat_features = ['season', 'weathersit', 'weekday']
    dummy = pd.get_dummies(data, columns=cat_features)

    permuts = 5
    metrics = pd.DataFrame()

    for i in range(permuts):
        rng = np.random.default_rng(i)
        fs = rng.permutation(features)

        preds, order_time = kfold_cv_progressive(dummy, 'instant', 'lin_reg', fs, 'cnt', 10)
        metric = get_predict_metrics(preds, 'cnt', order_time)
        metric['batch'] = i
        metrics = pd.concat([metrics, metric], axis=0, ignore_index=True)

    print(metrics)