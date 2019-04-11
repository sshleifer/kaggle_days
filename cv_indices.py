import pandas as pd
import pickle
import numpy as np


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# def use_splits():
#     for tr_idx in pickle_load('tr_indices.pkl'):
#         tr, val = train.loc[tr_idx], train.drop(tr_idx)
#
#     return

def predict_proba_fn_binary(m, x):
    return m.predict_proba(x)[:, 1]


def predict_proba(m, x):
    return m.predict_proba(x)


def shape_assert(a, b):
    message = f'a.shape = {a.shape[0]} but b.shape = {b.shape[0]}'
    assert a.shape[0] == b.shape[0], message


def cross_val_predict_proba_df(m, X, y, test_X, n_splits=5, binary=True, sample_weight=None,
                               stratified=False):
    shape_assert(X, y)
    predict_proba_fn = predict_proba_fn_binary if binary else predict_proba
    if binary:
        preds = pd.Series({k: np.nan for k in X.index})
    else:
        n_classes = len(np.unique(y))
        preds = pd.DataFrame({k: [np.nan] * n_classes for k in X.index}).T

    all_indices = X.index
    test_preds = []

    for tr_idx in pickle_load('tr_indices.pkl'):
        test_idx = all_indices.drop(tr_idx)
        if sample_weight is None:
            m.fit(X.iloc[tr_idx, :], y.iloc[tr_idx])
        else:
            m.fit(X.iloc[tr_idx, :], y.iloc[tr_idx], sample_weight=sample_weight.iloc[tr_idx])
        preds.iloc[test_idx] = predict_proba_fn(m, X.iloc[test_idx, :])
        test_preds.append(predict_proba_fn(test_X))
    return preds, test_preds
