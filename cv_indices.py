import pandas as pd
import pickle
import numpy as np
from tqdm import *

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


def cross_val_predict_proba_df(m, X, y, test_X, binary=True, sample_weight=None,
                               cv_path='tr_indices.pkl'):
    shape_assert(X, y)
    predict_proba_fn = predict_proba_fn_binary if binary else predict_proba
    if binary:
        preds = pd.Series({k: np.nan for k in X.index})
    else:
        n_classes = len(np.unique(y))
        preds = pd.DataFrame({k: [np.nan] * n_classes for k in X.index}).T

    all_indices = X.index
    test_preds = []
    for tr_idx in pickle_load(cv_path):
        test_idx = all_indices.drop(tr_idx)
        if sample_weight is None:
            m.fit(X.iloc[tr_idx, :], y.iloc[tr_idx])
        else:
            m.fit(X.iloc[tr_idx, :], y.iloc[tr_idx], sample_weight=sample_weight.iloc[tr_idx])
        preds.iloc[test_idx] = predict_proba_fn(m, X.iloc[test_idx, :])
        test_preds.append(predict_proba_fn(m, test_X))
    return preds, test_preds

from fastai.tabular import *
from fastai.callbacks import *
def cross_val_predict_fastai(train, cat_names, cont_names, test_list, procs,allowed_features, dep_var='target', ):
    all_te_probas = []

    all_indices = train.index
    val_preds = pd.Series({k: np.nan for k in X.index})
    for i, tr_idx in tqdm_notebook(list(enumerate(pickle_load('tr_indices.pkl')))):
        val_idx = all_indices.drop(tr_idx)
        _cat_names = set(cat_names).intersection(allowed_features)
        _cont_names = set(cat_names).intersection(allowed_features)
        data = (TabularList.from_df(
            train, path='train.csv',
            cat_names=cat_names, cont_names=_cont_names, procs=procs)
                .split_by_idx(val_idx)
                .label_from_df(cols=dep_var)
                .add_test(test_list)
                .databunch(bs=1024))
        learn = tabular_learner(data, layers=[200, 100], metrics=accuracy)
        save_name = f'best_classif_fold_{i}'
        callbacks = [
            CSVLogger(learn, filename='csv_logs_for_{i}'),
            SaveModelCallback(learn, name=save_name),
            EarlyStoppingCallback(learn, patience=2),
            GradientClipping(learn, 1.),
        ]
        learn.fit_one_cycle(
            8, .03, callbacks=callbacks
        )
        learn = learn.load(save_name)
        val_probas = (learn.get_preds(ds_type=DatasetType.Valid)[0][:, 1].cpu().numpy())
        val_preds.iloc[val_idx] = val_probas
        te_probas = (learn.get_preds(ds_type=DatasetType.Test)[0][:, 1].cpu().numpy())
        all_te_probas.append(te_probas)

