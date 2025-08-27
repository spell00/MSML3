import copy
import json
import os
from contextlib import suppress
try:
    from dvclive import Live
except Exception:
    Live = None
from contextlib import suppress
import neptune
from sklearn.calibration import calibration_curve
import pickle
import numpy as np
import pandas as pd
import datetime
from .utils import remove_zero_cols, scale_data, get_empty_lists
from .torch_utils import augment_data
from .loggings import log_ord, log_fct, save_confusion_matrix, log_neptune, plot_bars
from .sklearn_train_nocv import get_confusion_matrix, plot_roc
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
# from sklearn.multiclass import OneVsRestClassifier
try:
    from sklearn.multiclass import OneVsRestClassifier
except Exception:
    OneVsRestClassifier = None
# import mattews correlation from sklearn
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
# from scipy import stats
from .log_shap import log_shap
import xgboost
import matplotlib.pyplot as plt
# import pipeline from sklearn
from .utils import columns_stats_over0

import sys

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'

# DVC integration helpers
def _run_cmd_silent(cmd: str):
    try:
        return os.popen(cmd + ' 2>/dev/null').read().strip()
    except Exception:
        return ''

def dvc_available():
    return bool(_run_cmd_silent('command -v dvc'))

def dvc_add(path: str):
    if not dvc_available() or not os.path.exists(path):
        return False
    # Only add if metafile missing or mtime changed (cheap heuristic)
    meta = path + '.dvc'
    if os.path.isdir(path) and not os.path.exists(meta):
        os.system(f'dvc add "{path}" >/dev/null 2>&1')
        return True
    if os.path.isfile(path) and not os.path.exists(meta):
        os.system(f'dvc add "{path}" >/dev/null 2>&1')
        return True
    return False

def dvc_push():
    if not dvc_available():
        return False
    os.system('dvc push >/dev/null 2>&1')
    return True


# Convert numpy types to native Python types
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def get_size_in_mb(obj):
    size_in_bytes = sys.getsizeof(obj)
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb


def score(predictions, labels):
    return ACC(labels, predictions)


def make_params_grid(h_params):
    hparams = {}
    param_grid = {}
    features_cutoff = None
    threshold = None
    scaler_name = None
    n_aug = 0
    for name, param in zip(h_params.keys(), h_params.values()):
        hparams[name] = param
        if name == 'features_cutoff':
            features_cutoff = param
        elif name == 'threshold':
            threshold = param
        elif name == 'scaler':
            scaler_name = param
        elif name == 'n_aug':
            n_aug = param
        elif name == 'p':
            p = param
        elif name == 'g':
            g = param
        else:
            param_grid[name] = param
    hparams['threshold'] = param_grid['threshold'] = threshold = 0
    return hparams, param_grid, features_cutoff, threshold, scaler_name, n_aug, p, g


def add_infos_to_dict(data_dict, lists, unique_labels):
    lists['inds']['train'] += [data_dict['inds']['train']]
    lists['inds']['valid'] += [data_dict['inds']['valid']]
    lists['inds']['test'] += [data_dict['inds']['test']]

    lists['names']['train'] += [data_dict['names']['train']]
    lists['names']['valid'] += [data_dict['names']['valid']]
    lists['names']['test'] += [data_dict['names']['test']]

    lists['batches']['train'] += [data_dict['batches']['train']]
    lists['batches']['valid'] += [data_dict['batches']['valid']]
    lists['batches']['test'] += [data_dict['batches']['test']]
    lists['unique_batches']['train'] += [list(np.unique(data_dict['batches']['train']))]
    lists['unique_batches']['valid'] += [list(np.unique(data_dict['batches']['valid']))]
    lists['unique_batches']['test'] += [list(np.unique(data_dict['batches']['test']))]
    lists['classes']['train'] += [np.array([
        np.argwhere(label == unique_labels)[0][0] for label in data_dict['labels']['train']
    ])]
    lists['classes']['valid'] += [np.array([
        np.argwhere(label == unique_labels)[0][0] for label in data_dict['labels']['valid']
    ])]
    lists['classes']['test'] += [np.array([
        np.argwhere(label == unique_labels)[0][0] for label in data_dict['labels']['test']
    ])]
    lists['labels']['train'] += [data_dict['labels']['train']]
    lists['labels']['valid'] += [data_dict['labels']['valid']]
    lists['labels']['test'] += [data_dict['labels']['test']]
    return lists


def augment_dataset(data_dict, n_aug, p, g):
    if n_aug > 0:
        data_dict['data']['train'] = data_dict['data']['train'].fillna(0)
        data_dict['data']['train'] = augment_data(data_dict['data']['train'], n_aug, p, g)
        data_dict['labels']['train'] = np.concatenate([data_dict['labels']['train']] * (n_aug + 1))
        data_dict['batches']['train'] = np.concatenate([data_dict['batches']['train']] * (n_aug + 1))
    else:
        data_dict['data']['train'] = data_dict['data']['train'].fillna(0)
    data_dict['data']['valid'] = data_dict['data']['valid'].fillna(0)
    data_dict['data']['test'] = data_dict['data']['test'].fillna(0)
    return data_dict


def get_dmatrices(args, data_dict, lists):
    if args.sparse_matrix:
        dtrain = xgboost.DMatrix(
            data_dict['data']['train'].astype(pd.SparseDtype("float", 0)),
            label=lists['classes']['train'][-1]
        )
        dvalid = xgboost.DMatrix(
            data_dict['data']['valid'].astype(pd.SparseDtype("float", 0)),
            label=lists['classes']['valid'][-1]
        )
        dtest = xgboost.DMatrix(
            data_dict['data']['test'].astype(pd.SparseDtype("float", 0)),
            label=lists['classes']['test'][-1]
        )
    else:
        dtrain = xgboost.DMatrix(data_dict['data']['train'], label=lists['classes']['train'][-1])
        dvalid = xgboost.DMatrix(data_dict['data']['valid'], label=lists['classes']['valid'][-1])
        dtest = xgboost.DMatrix(data_dict['data']['test'], label=lists['classes']['test'][-1])
    # Check for NaNs in the DMatrix
    try:
        assert not np.isnan(dtrain.get_label()).any()
        assert not np.isnan(dvalid.get_label()).any()
        assert not np.isnan(dtest.get_label()).any()
    except Exception as e:
        print(f"Error in DMatrix: {e}")
        exit

    return {
        'train': dtrain,
        'valid': dvalid,
        'test': dtest
    }


def add_results_to_list(m, upos, all_data, lists, dmatrices, data_dict, unique_labels):
    # TODO reduce list of variables
    preds = {}
    preds['train'] = m.predict(dmatrices['train']).argmax(axis=1)
    preds['valid'] = m.predict(dmatrices['valid']).argmax(axis=1)
    preds['test'] = m.predict(dmatrices['test']).argmax(axis=1)

    lists['acc']['train'] += [ACC(preds['train'], lists['classes']['train'][-1])]
    lists['preds']['train'] += [preds['train']]

    lists['acc']['valid'] += [ACC(preds['valid'], lists['classes']['valid'][-1])]
    lists['acc']['test'] += [ACC(preds['test'], lists['classes']['test'][-1])]
    lists['preds']['valid'] += [preds['valid']]
    lists['preds']['test'] += [preds['test']]
    lists['mcc']['train'] += [MCC(lists['classes']['train'][-1], lists['preds']['train'][-1])]
    lists['mcc']['valid'] += [MCC(lists['classes']['valid'][-1], lists['preds']['valid'][-1])]
    lists['mcc']['test'] += [MCC(lists['classes']['test'][-1], lists['preds']['test'][-1])]

    # TODO all_data necessary? replace by upos['inputs']
    if all_data['inputs']['urinespositives'].shape[0] > 0:
        durines = xgboost.DMatrix(upos['inputs'])
        lists['preds']['posurines'] += [m.predict(durines).argmax(axis=1)]
        lists['labels']['posurines'] += [upos['labels']]
        lists['classes']['posurines'] += [
            np.array([np.argwhere(label == unique_labels)[0][0] for label in upos['labels']])
        ]
        lists['names']['posurines'] += [upos['names']]
        lists['batches']['posurines'] += [upos['batches']]
        try:
            lists['proba']['posurines'] += [m.predict(durines)]
        except Exception as e:
            print(f"Error in saving proba: {e}")
            pass
        # lists['mcc']['posurines'] += [MCC(lists['classes']['posurines'][-1], lists['preds']['posurines'][-1])]
        # lists['acc']['posurines'] += [ACC(lists['preds']['posurines'][-1], lists['classes']['posurines'][-1])]

    try:
        lists['proba']['train'] += [m.predict(dmatrices['train'])]
        lists['proba']['valid'] += [m.predict(dmatrices['valid'])]
        lists['proba']['test'] += [m.predict(dmatrices['test'])]
        data_list = {
            'valid': {
                'inputs': data_dict['data']['valid'],
                'labels': lists['classes']['valid'][-1],
                'preds': lists['preds']['valid'][-1],
                'proba': lists['proba']['valid'][-1],
                'batches': data_dict['batches']['valid'],
                'names': all_data['names']['valid']
            },
            'test': {
                'inputs': data_dict['data']['test'],
                'labels': lists['classes']['test'][-1],
                'preds': lists['preds']['test'][-1],
                'proba': lists['proba']['test'][-1],
                'batches': data_dict['batches']['test'],
                'names': all_data['names']['test']
            }
        }
    except Exception as e:
        print(f"Error in add_results_to_list prediction: {e}")
        data_list = {
            'valid': {
                'inputs': data_dict['data']['valid'],
                'labels': lists['classes']['valid'][-1],
                'preds': lists['preds']['valid'][-1],
                'batches': data_dict['batches']['valid'],
                'names': lists['names']['valid'][-1]
            },
            'test': {
                'inputs': data_dict['data']['test'],
                'labels': lists['classes']['test'][-1],
                'preds': lists['preds']['test'][-1],
                'batches': data_dict['batches']['test'],
                'names': lists['names']['test'][-1]
            }
        }
    return lists, data_list


class Train:
    def __init__(self, name, model, data, uniques,
                 log_path, args, logger,
                 binary_path=None):
        # self.hparams_names = None
        self.best_roc_score = -1
        self.args = args
        self.log_path = log_path
        self.binary_path = binary_path
        self.model = model
        self.data = data
        self.logger = logger
        self.best_scores = {
            m: {
                'train': -1,
                'valid': -1,
                'test': -1,
                'posurines': -1,
            } for m in ['acc', 'mcc', 'tpr', 'tnr', 'precision']

        }
        if binary_path:
            self.best_scores['bact_acc'] = {
                'train': -1,
                'valid': -1,
                'test': -1,
                'posurines': -1,
            }
            self.best_scores['bact_mcc'] = {
                'train': -1,
                'valid': -1,
                'test': -1,
                'posurines': -1,
            }

        self.iter = 0
        self.model = model
        self.name = name
        # dvclive setup
        self.live = None
        self.uniques = uniques
        self.uniques2 = copy.deepcopy(self.uniques)
        self.uniques2['labels'] = None
        self.best_params_dict = {}
        self.best_params_dict_values = {}

        if self.args.mz < 1:
            mz_rounding = len(str(self.args.mzp).split('.')[-1]) + 1
        else:
            mz_rounding = 1

        if self.args.rt < 1:
            rt_rounding = len(str(self.args.rtp).split('.')[-1]) + 1
        else:
            rt_rounding = 1
        self.bins = {
                'mz_max': self.args.max_mz,
                'mz_min': self.args.min_mz,
                'rt_max': self.args.max_rt,
                'rt_min': self.args.min_rt,
                'mz_rounding': mz_rounding,
                'rt_rounding': rt_rounding,
                'mz_bin': self.args.mz,
                'rt_bin': self.args.rt,
            }

    def split_data(self, all_data, upos, h):
        # TODO no upos should be passed, should be in all_data
        if self.args.train_on == 'all':
            if self.args.groupkfold:
                # skf = GroupShuffleSplit(n_splits=5, random_state=seed)
                skf = StratifiedGroupKFold(
                    n_splits=len(np.unique(all_data['batches']['all'])), shuffle=True, random_state=42
                )
                train_nums = np.arange(0, len(all_data['labels']['all']))
                splitter = skf.split(train_nums, all_data['labels']['all'], all_data['batches']['all'])
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                train_nums = np.arange(0, len(all_data['labels']['all']))
                splitter = skf.split(train_nums, all_data['labels']['all'], all_data['labels']['all'])
                # Split urinespositives so that equivalent number of urines are in each split
                # inds_to_keep = np.where(np.isin(all_data['labels']['urinespositives'], ['eco', 'kpn', 'sag']))[0]

                skf_upos = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                train_nums_upos = np.arange(0, len(all_data['labels']['urinespositives']))
                splitter_upos = skf_upos.split(train_nums_upos, all_data['labels']['urinespositives'])

            if h > 0 and h < self.args.n_repeats - 1:
                for i in range(h):
                    _, _ = splitter.__next__()
                _, valid_inds = splitter.__next__()
                _, test_inds = splitter.__next__()
                if not self.args.groupkfold:
                    for i in range(h):
                        _, _ = splitter_upos.__next__()
                    _, valid_inds_upos = splitter_upos.__next__()
                    _, test_inds_upos = splitter_upos.__next__()

            elif h == self.args.n_repeats - 1:
                _, test_inds = splitter.__next__()
                for i in range(h-1):
                    _, _ = splitter.__next__()
                _, valid_inds = splitter.__next__()
                if not self.args.groupkfold:
                    _, test_inds_upos = splitter_upos.__next__()
                    for i in range(h-1):
                        _, _ = splitter_upos.__next__()
                    _, valid_inds_upos = splitter_upos.__next__()
            else:
                _, valid_inds = splitter.__next__()
                _, test_inds = splitter.__next__()
                if not self.args.groupkfold:
                    _, valid_inds_upos = splitter_upos.__next__()
                    _, test_inds_upos = splitter_upos.__next__()

            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            train_data, valid_data, test_data =\
                all_data['inputs']['all'].iloc[train_inds], \
                all_data['inputs']['all'].iloc[valid_inds], \
                all_data['inputs']['all'].iloc[test_inds]
            train_labels, valid_labels, test_labels =\
                all_data['labels']['all'][train_inds], \
                all_data['labels']['all'][valid_inds], \
                all_data['labels']['all'][test_inds]
            train_batches, valid_batches, test_batches =\
                all_data['batches']['all'][train_inds], \
                all_data['batches']['all'][valid_inds], \
                all_data['batches']['all'][test_inds]
            train_batches_labels, valid_batches_labels, test_batches_labels =\
                all_data['batches_labels']['all'][train_inds], \
                all_data['batches_labels']['all'][valid_inds], \
                all_data['batches_labels']['all'][test_inds]
            train_names, valid_names, test_names =\
                all_data['names']['all'][train_inds], \
                all_data['names']['all'][valid_inds], \
                all_data['names']['all'][test_inds]

            if not self.args.groupkfold:
                train_inds_upos = [
                    x for x in train_nums_upos if x not in np.concatenate((valid_inds_upos, test_inds_upos))
                ]
                train_data, valid_data = pd.concat((train_data, upos['data'].iloc[train_inds_upos]), axis=0), \
                    pd.concat((valid_data, upos['data'].iloc[valid_inds_upos]), axis=0)
                train_labels, valid_labels = np.concatenate((train_labels, upos['labels'][train_inds_upos])), \
                    np.concatenate((valid_labels, upos['labels'][valid_inds_upos]))
                train_batches, valid_batches = np.concatenate((train_batches, upos['batches'][train_inds_upos])), \
                    np.concatenate((valid_batches, upos['batches'][valid_inds_upos]))
                train_names, valid_names = np.concatenate((train_names, upos['names'][train_inds_upos])), \
                    np.concatenate((valid_names, upos['names'][valid_inds_upos]))

                # drop urinespositives from train and valid
                b_inds = np.concatenate((train_inds_upos, valid_inds_upos)).astype(int)
                upos['labels'] = np.delete(upos['labels'], b_inds)
                upos['batches'] = np.delete(upos['batches'], b_inds)
                upos['names'] = np.delete(upos['names'], b_inds)
                upos['data'] = upos['data'].loc[upos['names']]
                upos['labels'] = np.delete(upos['labels'], b_inds)
            # remove labels that are not in train
            unique_train_labels = np.unique(train_labels)
            valid_to_keep = np.array(
                [i for i, l in enumerate(valid_labels) if l in unique_train_labels]
            )
            test_to_keep = np.array(
                [i for i, l in enumerate(test_labels) if l in unique_train_labels]
            )

            valid_data, valid_labels, valid_batches, valid_names =\
                valid_data.iloc[valid_to_keep], \
                valid_labels[valid_to_keep], \
                valid_batches[valid_to_keep], \
                valid_names[valid_to_keep]
            test_data, test_labels, test_batches, test_names =\
                test_data.iloc[test_to_keep], \
                test_labels[test_to_keep], \
                test_batches[test_to_keep], \
                test_names[test_to_keep]
            # valid_inds, test_inds = valid_inds[valid_to_keep], test_inds[test_to_keep]
        elif self.args.train_on == 'all_lows':
            # keep all concs for train to be all_data['concs']['all'] == 'l'
            train_inds = np.argwhere(all_data['concs']['all'] == 'l').flatten()
            valid_inds = np.argwhere(all_data['concs']['all'] == 'h').flatten()
            blanc_inds = np.argwhere(all_data['concs']['all'] == 'na').flatten()
            train_data, valid_data, blanc_data =\
                all_data['inputs']['all'].iloc[train_inds], \
                all_data['inputs']['all'].iloc[valid_inds], \
                all_data['inputs']['all'].iloc[blanc_inds]
            train_labels, valid_labels, blanc_labels =\
                all_data['labels']['all'][train_inds], \
                all_data['labels']['all'][valid_inds], \
                all_data['labels']['all'][blanc_inds]
            train_batches, valid_batches, blanc_batches =\
                all_data['batches']['all'][train_inds], \
                all_data['batches']['all'][valid_inds], \
                all_data['batches']['all'][blanc_inds]
            train_names, valid_names, blanc_names =\
                all_data['names']['all'][train_inds], \
                all_data['names']['all'][valid_inds], \
                all_data['names']['all'][blanc_inds]
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            blanc_nums = np.arange(0, len(blanc_labels))
            splitter = skf.split(blanc_nums, blanc_labels, blanc_batches)
            blanc_train_inds, blanc_valid_inds = splitter.__next__()
            train_data, valid_data = pd.concat((train_data, blanc_data.iloc[blanc_train_inds])), \
                pd.concat((valid_data, blanc_data.iloc[blanc_valid_inds]))
            train_labels, valid_labels = np.concatenate((train_labels, blanc_labels[blanc_train_inds])), \
                np.concatenate((valid_labels, blanc_labels[blanc_valid_inds]))
            train_batches, valid_batches = np.concatenate((train_batches, blanc_batches[blanc_train_inds])), \
                np.concatenate((valid_batches, blanc_batches[blanc_valid_inds]))
            train_names, valid_names = np.concatenate((train_names, blanc_names[blanc_train_inds])), \
                np.concatenate((valid_names, blanc_names[blanc_valid_inds]))

            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            valid_nums = np.arange(0, len(valid_labels))
            splitter = skf.split(valid_nums, valid_labels, valid_batches)
            test_inds, valid_inds = splitter.__next__()
            test_data, valid_data = valid_data.iloc[test_inds], valid_data.iloc[valid_inds]
            test_labels, valid_labels = valid_labels[test_inds], valid_labels[valid_inds]
            test_batches, valid_batches = valid_batches[test_inds], valid_batches[valid_inds]
            test_names, valid_names = valid_names[test_inds], valid_names[valid_inds]
        elif self.args.train_on == 'all_highs':
            # keep all concs for train to be all_data['concs']['all'] == 'h'
            train_inds = np.argwhere(all_data['concs']['all'] == 'h').flatten()
            valid_inds = np.argwhere(all_data['concs']['all'] == 'l').flatten()
            blanc_inds = np.argwhere(all_data['concs']['all'] == 'na').flatten()
            train_data, valid_data, blanc_data =\
                all_data['inputs']['all'].iloc[train_inds], \
                all_data['inputs']['all'].iloc[valid_inds], \
                all_data['inputs']['all'].iloc[blanc_inds]
            train_labels, valid_labels, blanc_labels =\
                all_data['labels']['all'][train_inds], \
                all_data['labels']['all'][valid_inds], \
                all_data['labels']['all'][blanc_inds]
            train_batches, valid_batches, blanc_batches =\
                all_data['batches']['all'][train_inds], \
                all_data['batches']['all'][valid_inds], \
                all_data['batches']['all'][blanc_inds]
            train_names, valid_names, blanc_names =\
                all_data['names']['all'][train_inds], \
                all_data['names']['all'][valid_inds], \
                all_data['names']['all'][blanc_inds]

            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            blanc_nums = np.arange(0, len(blanc_labels))
            splitter = skf.split(blanc_nums, blanc_batches, blanc_batches)
            blanc_train_inds, blanc_valid_inds = splitter.__next__()
            train_data, valid_data = pd.concat((train_data, blanc_data.iloc[blanc_train_inds])), \
                pd.concat((valid_data, blanc_data.iloc[blanc_valid_inds]))
            train_labels, valid_labels = np.concatenate((train_labels, blanc_labels[blanc_train_inds])), \
                np.concatenate((valid_labels, blanc_labels[blanc_valid_inds]))
            train_batches, valid_batches = np.concatenate((train_batches, blanc_batches[blanc_train_inds])), \
                np.concatenate((valid_batches, blanc_batches[blanc_valid_inds]))
            train_names, valid_names = np.concatenate((train_names, blanc_names[blanc_train_inds])), \
                np.concatenate((valid_names, blanc_names[blanc_valid_inds]))

            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            valid_nums = np.arange(0, len(valid_labels))
            splitter = skf.split(valid_nums, valid_batches, valid_batches)
            test_inds, valid_inds = splitter.__next__()
            test_data, valid_data = valid_data.iloc[test_inds], valid_data.iloc[valid_inds]
            test_labels, valid_labels = valid_labels[test_inds], valid_labels[valid_inds]
            test_batches, valid_batches = valid_batches[test_inds], valid_batches[valid_inds]
            test_names, valid_names = valid_names[test_inds], valid_names[valid_inds]

        data_dict = {
            'data': {
                'train': train_data,
                'valid': valid_data,
                'test': test_data,
            },
            'labels': {
                'train': train_labels,
                'valid': valid_labels,
                'test': test_labels,
            },
            'batches': {
                'train': train_batches,
                'valid': valid_batches,
                'test': test_batches,
            },
            'batches_labels': {
                'train': train_batches_labels,
                'valid': valid_batches_labels,
                'test': test_batches_labels,
            },
            'names': {
                'train': train_names,
                'valid': valid_names,
                'test': test_names,
            },
            'inds': {
                'train': train_inds,
                'valid': valid_inds,
                'test': test_inds,
            }
        }

        return data_dict

    def log_shap(self, m, lists, data_dict, run, mlops):
        if self.args.log_shap:
            Xs = {
                'train': data_dict['data']['train'],
                'valid': data_dict['data']['valid'],
                'test': data_dict['data']['test'],
                # 'posurines': all_data['inputs']['urinespositives'],
            }
            ys = {
                'train': lists['classes']['train'][-1],
                'valid': lists['classes']['valid'][-1],
                'test': lists['classes']['test'][-1],
                # 'posurines': np.array([
                #     np.argwhere(
                #         label == self.unique_labels)[0][0]
                #     for label in all_data['labels']['urinespositives']
                # ]),
            }
            labels = {
                'train': lists['labels']['train'][-1],
                'valid': lists['labels']['valid'][-1],
                'test': lists['labels']['test'][-1],
                # 'posurines': np.array([
                #     np.argwhere(label == self.unique_labels)[0][0]
                #     for label in all_data['labels']['urinespositives']
                # ]),
            }
            args_dict = {
                'inputs': Xs,
                'ys': ys,
                'labels': labels,
                'model': m,
                'model_name': self.args.model_name,
                'log_path': self.log_path,
            }
            if mlops == 'neptune':
                run = log_shap(run, args_dict, 'neptune')
            if mlops == 'dvclive':
                run = log_shap(run, args_dict, 'dvclive')
        else:
            if mlops == 'neptune':
                run['log_shap'] = 0
            if mlops == 'dvclive':
                run.log({'log_shap': 0})

    def upos_operations(self, all_data):
        # Basically just takes the samples back into training
        if all_data['inputs']['urinespositives'].shape[0] > 0:
            # Keep only eco
            from collections import Counter
            print(Counter(all_data['labels']['urinespositives']))
            print("Batch counter:", Counter(all_data['batches_labels']['urinespositives']))
            
            # Counter of labels per batch
            print("Labels per batch:")
            for batch in np.unique(all_data['batches_labels']['urinespositives']):
                batch_mask = all_data['batches_labels']['urinespositives'] == batch
                batch_labels = all_data['labels']['urinespositives'][batch_mask]
                print(f"  Batch {batch}: {dict(Counter(batch_labels))}")
            
            bacts = self.args.patient_bact.split(',')
            if bacts[0] != '':
                inds_to_keep = np.argwhere(
                    np.isin(
                        all_data['labels']['urinespositives'],
                        ['eco', 'kpn', 'sag']
                    )
                ).flatten()
            else:
                inds_to_keep = np.arange(len(all_data['labels']['urinespositives']))

            for k in all_data.keys():
                if k == 'inputs':
                    pass
                elif k == 'sets':
                    all_data[k]['urinespositives'] = []
                else:
                    all_data[k]['urinespositives'] = all_data[k]['urinespositives'][inds_to_keep]
            all_data['inputs']['urinespositives'] =\
                all_data['inputs']['urinespositives'].loc[all_data['names']['urinespositives'], :]
            print(Counter(all_data['labels']['urinespositives']))
            if self.args.groupkfold:
                b10_inds = np.argwhere(
                    all_data['batches']['urinespositives'] == np.argwhere(self.uniques['batches'] == 'b10')[0][0]
                ).flatten()
                bpatients = np.argwhere(
                    all_data['batches']['urinespositives'] == np.argwhere(self.uniques['batches'] == 'bpatients')[0][0]
                ).flatten()
                b_inds = np.concatenate((b10_inds, bpatients))
                urines10 = np.array([x.split('_')[-2] for x in all_data['names']['urinespositives'][b_inds]])
                # Make sure the labels are the good ones in the names and inputs labels
                all_data['inputs']['all'] =\
                    pd.concat((all_data['inputs']['all'], all_data['inputs']['urinespositives'].iloc[b_inds]))
                all_data['labels']['all'] =\
                    np.concatenate((all_data['labels']['all'], all_data['labels']['urinespositives'][b_inds]))
                all_data['batches']['all'] =\
                    np.concatenate((all_data['batches']['all'], all_data['batches']['urinespositives'][b_inds]))
                all_data['batches_labels']['all'] =\
                    np.concatenate((all_data['batches_labels']['all'], all_data['batches_labels']['urinespositives'][b_inds]))
                all_data['names']['all'] =\
                    np.concatenate((all_data['names']['all'], all_data['names']['urinespositives'][b_inds]))
                all_data['concs']['all'] =\
                    np.concatenate((all_data['concs']['all'], all_data['concs']['urinespositives'][b_inds]))

                # Remove b10 and bpatients from urinespositives
                all_data['inputs']['urinespositives'] =\
                    all_data['inputs']['urinespositives'].drop(all_data['inputs']['urinespositives'].index[b_inds])
                all_data['labels']['urinespositives'] = np.delete(all_data['labels']['urinespositives'], b_inds)
                all_data['batches']['urinespositives'] = np.delete(all_data['batches']['urinespositives'], b_inds)
                all_data['batches_labels']['urinespositives'] = np.delete(all_data['batches_labels']['urinespositives'], b_inds)
                all_data['names']['urinespositives'] = np.delete(all_data['names']['urinespositives'], b_inds)
                all_data['concs']['urinespositives'] = np.delete(all_data['concs']['urinespositives'], b_inds)
                # Remove urines10 from urinespositives because duplicated
                inds_to_keep = np.array(
                    [i for i, x in enumerate(all_data['names']['urinespositives']) if x.split('_')[-2] not in urines10]
                )
                all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'].iloc[inds_to_keep]
                all_data['labels']['urinespositives'] = all_data['labels']['urinespositives'][inds_to_keep]
                all_data['batches']['urinespositives'] = all_data['batches']['urinespositives'][inds_to_keep]
                all_data['batches_labels']['urinespositives'] = all_data['batches_labels']['urinespositives'][inds_to_keep]
                all_data['names']['urinespositives'] = all_data['names']['urinespositives'][inds_to_keep]
                all_data['concs']['urinespositives'] = all_data['concs']['urinespositives'][inds_to_keep]

        return all_data

    def train(self, h_params):
        metrics = {}
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        scaler_name = 'none'
        hparams = {}
        n_aug = 0
        for name, param in zip(self.hparams_names, h_params):
            hparams[name] = param
            if name == 'features_cutoff':
                features_cutoff = param
            elif name == 'threshold':
                threshold = param
            elif name == 'scaler':
                scaler_name = param
            elif name == 'n_aug':
                n_aug = param
            elif name == 'p':
                p = param
            elif name == 'g':
                g = param
            else:
                param_grid[name] = param
        # n_aug = 0  # TODO TO REMOVE
        hparams['threshold'] = param_grid['threshold'] = threshold = 0
        other_params = {
            'p': p, 
            'g': g, 
            'n_aug': n_aug,
            'features_cutoff': features_cutoff,
            'threshold': threshold,

        }

        lists = get_empty_lists()

        all_data = {
            'inputs': copy.deepcopy(self.data['inputs']),
            'labels': copy.deepcopy(self.data['labels']),
            'batches': copy.deepcopy(self.data['batches']),
            'batches_labels': copy.deepcopy(self.data['batches_labels']),
            'concs': copy.deepcopy(self.data['concs']),
            'names': copy.deepcopy(self.data['names']),
            'manips': copy.deepcopy(self.data['manips']),
            'urines': copy.deepcopy(self.data['urines']),
            'cats': copy.deepcopy(self.data['cats']),
        }

        # Exclude user-specified batches (substrings, case-insensitive)
        exclude_patterns = set(getattr(self.args, 'exclude_batches', []) or [])
        def _filter_split(split_name):
            if split_name not in all_data['batches_labels']:
                return
            batches_arr = np.array(all_data['batches_labels'][split_name])
            mask = np.array([
                (str(b).lower() not in exclude_patterns and
                 not any(pat in str(b).lower() for pat in exclude_patterns))
                for b in batches_arr
            ])
            if split_name in all_data['inputs']:
                try:
                    all_data['inputs'][split_name] = all_data['inputs'][split_name].iloc[mask, :]
                except Exception:
                    pass
            for key in ['labels', 'batches', 'batches_labels', 'concs', 'names', 'manips', 'urines', 'cats']:
                if split_name in all_data.get(key, {}):
                    try:
                        all_data[key][split_name] = np.array(all_data[key][split_name])[mask]
                    except Exception:
                        try:
                            all_data[key][split_name] = [all_data[key][split_name][i] for i, keep in enumerate(mask) if keep]
                        except Exception:
                            pass
        for split in list(all_data['batches_labels'].keys()):
            _filter_split(split)

        if self.args.binary:
            all_data['labels']['all'] = np.array(['blanc' if label == 'blanc' else 'bact' for label in all_data['labels']['all']])
        # self.unique_labels devrait disparaitre et remplace par self.uniques['labels']
        self.unique_labels = np.array(np.unique(all_data['labels']['all']))
        # place blancs at the end
        blanc_class = np.argwhere(self.unique_labels == 'blanc').flatten()[0]
        self.unique_labels = np.concatenate((['blanc'], np.delete(self.unique_labels, blanc_class)))
        self.model_name = f'binary{self.args.binary}_{self.args.model_name}'
        self.uniques['labels'] = self.unique_labels

        if threshold > 0:
            not_zeros_col = remove_zero_cols(all_data['inputs']['all'], threshold)
            all_data['inputs']['all'] = all_data['inputs']['all'].iloc[:, not_zeros_col]
            all_data['inputs']['train'] = all_data['inputs']['train'].iloc[:, not_zeros_col]
            all_data['inputs']['valid'] = all_data['inputs']['valid'].iloc[:, not_zeros_col]
            all_data['inputs']['test'] = all_data['inputs']['test'].iloc[:, not_zeros_col]
            all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'].iloc[:, not_zeros_col]

        if self.args.log_neptune:
            # Create a Neptune run object
            run = neptune.init_run(
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
                source_files=['train.py',
                              'dataset.py',
                              'utils.py',
                              'sklearn_train_nocv.py',
                              'loggings.py',
                              'metrics.py',
                              "**/*.py"
                              ],
            )  # your credentials
            model = neptune.init_model_version(
                model=f'{NEPTUNE_MODEL_NAME}{self.args.model_name.upper()}{self.args.binary}',
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
            )
            # Keep references for later artifact logging
            self.model_version_ref = model
            self.run_ref = run
            model['hparams'] = run["hparams"] = hparams
            model["csv_file"] = run["csv_file"] = self.args.csv_file
            model["model_name"] = run["model_name"] = self.model_name
            model["groupkfold"] = run["groupkfold"] = self.args.groupkfold
            model["dataset_name"] = run["dataset_name"] = 'MSML-Bacteria'
            model["scaler_name"] = run["scaler_name"] = scaler_name
            model["mz_min"] = run["mz_min"] = self.args.min_mz
            model["mz_max"] = run["mz_max"] = self.args.max_mz
            model["rt_min"] = run["rt_min"] = self.args.min_rt
            model["rt_max"] = run["rt_max"] = self.args.max_rt
            model["mz_bin"] = run["mz_bin"] = self.args.mz
            model["rt_bin"] = run["rt_bin"] = self.args.rt
            model["path"] = run["path"] = self.log_path
            model["concs"] = run["concs"] = self.args.concs
            model["binary"] = run["binary"] = self.args.binary
            model['spd'] = run['spd'] = self.args.spd
            model['ovr'] = run['ovr'] = self.args.ovr
            model['train_on'] = run['train_on'] = self.args.train_on
            model['n_features'] = run['n_features'] = self.args.n_features
            model['total_features'] = run['total_features'] = all_data['inputs']['all'].shape[1]
            model['ms_level'] = run['ms_level'] = self.args.ms_level
            model['log'] = run['log'] = self.args.log
            model['batches'] = run['batches'] = '-'.join(self.uniques['batches'])
            model['context'] = run['context'] = 'train'
            model['remove_bad_samples'] = run['remove_bad_samples'] = self.args.remove_bad_samples
            # Additional context fields
            try:
                model['mode'] = run['mode'] = getattr(self.args, 'mode', '')
                model['features_selection'] = run['features_selection'] = getattr(self.args, 'features_selection', '')
                model['combat'] = run['combat'] = getattr(self.args, 'combat', '')
                model['shift'] = run['shift'] = getattr(self.args, 'shift', '')
                model['device'] = run['device'] = getattr(self.args, 'device', '')
                model['fp'] = run['fp'] = getattr(self.args, 'fp', '')
                model['scheduler'] = run['scheduler'] = getattr(self.args, 'scheduler', '')
                model['dloss'] = run['dloss'] = getattr(self.args, 'dloss', '')
                model['classif_loss'] = run['classif_loss'] = getattr(self.args, 'classif_loss', '')
                model['rec_loss'] = run['rec_loss'] = getattr(self.args, 'rec_loss', '')
                model['early_stop'] = run['early_stop'] = getattr(self.args, 'early_stop', '')
                model['early_warmup_stop'] = run['early_warmup_stop'] = getattr(self.args, 'early_warmup_stop', '')
                model['n_layers'] = run['n_layers'] = getattr(self.args, 'n_layers', '')
                model['ae_layers_max_neurons'] = run['ae_layers_max_neurons'] = getattr(self.args, 'ae_layers_max_neurons', '')
                model['n_repeats'] = run['n_repeats'] = getattr(self.args, 'n_repeats', '')
                model['threshold'] = run['threshold'] = param_grid.get('threshold', 0)
                model['features_cutoff'] = run['features_cutoff'] = param_grid.get('features_cutoff', '')
                model['exclude_batches'] = run['exclude_batches'] = ','.join(getattr(self.args, 'exclude_batches', []) or [])
                # Optional augmentation / regularization flags
                for flag in ['add_noise','normalize','use_mapping','use_sigmoid','use_threshold','use_smoothing','variational','kan','ovr']:
                    if hasattr(self.args, flag):
                        model[flag] = run[flag] = getattr(self.args, flag)
                for flag in ['max_norm','max_warmup','num_workers','clip_val','random_recs','rec_prototype','min_features_importance','patient_bact','colsample_bytree','max_bin','sparse_matrix','xgboost_features','prune_threshold']:
                    if hasattr(self.args, flag):
                        model[flag] = run[flag] = getattr(self.args, flag)
            except Exception as e:
                print('Neptune extra logging fields failed:', e)
        else:
            model = None
            run = None

        all_data, scaler = scale_data(scaler_name, all_data)

        infos = {
            'scaler': scaler_name,
            'h_params': param_grid,
            'mz': self.args.mz,
            'rt': self.args.rt,
            'mz_min': self.args.min_mz,
            'mz_max': self.args.max_mz,
            'rt_min': self.args.min_rt,
            'rt_max': self.args.max_rt,
            'mz_bin': self.args.mz,
            'rt_bin': self.args.rt,
            'features_cutoff': features_cutoff,
            'threshold': threshold,
            'inference': False,
        }

        columns_stats_over0(all_data['inputs']['all'], infos, False)

        # save scaler
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/scaler_{scaler_name}_tmp.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl', 'wb') as f:
            pickle.dump(all_data['inputs']['all'].columns, f)

        print(f'Iteration: {self.iter}')
        # models = []
        best_iteration = []
        h = 0

        if self.args.groupkfold:
            self.args.n_repeats = len(np.unique(all_data['batches']['all']))
        while h < self.args.n_repeats:                                
            lists['names']['posurines'] += [np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])]
            lists['batches']['posurines'] += [all_data['batches']['urinespositives']]
            if self.args.train_on == 'all':
                if self.args.groupkfold:
                    # skf = GroupShuffleSplit(n_splits=5, random_state=seed)
                    skf = StratifiedGroupKFold(n_splits=len(np.unique(all_data['batches']['all'])), shuffle=True, random_state=42)
                    train_nums = np.arange(0, len(all_data['labels']['all']))
                    splitter = skf.split(train_nums, self.data['labels']['all'], all_data['batches']['all'])
                else:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    train_nums = np.arange(0, len(all_data['labels']['all']))
                    splitter = skf.split(train_nums, self.data['labels']['all'], self.data['labels']['all'])
                if h > 0 and h < self.args.n_repeats - 1:
                    for i in range(h):
                        _, _ = splitter.__next__()
                    _, valid_inds = splitter.__next__()
                    _, test_inds = splitter.__next__()

                elif h == self.args.n_repeats - 1:
                    _, test_inds = splitter.__next__()
                    for i in range(h-1):
                        _, _ = splitter.__next__()
                    _, valid_inds = splitter.__next__()
                else:
                    _, valid_inds = splitter.__next__()
                    _, test_inds = splitter.__next__()

                train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

                train_data, valid_data, test_data =\
                      all_data['inputs']['all'].iloc[train_inds], all_data['inputs']['all'].iloc[valid_inds], all_data['inputs']['all'].iloc[test_inds]
                train_labels, valid_labels, test_labels =\
                      all_data['labels']['all'][train_inds], all_data['labels']['all'][valid_inds], all_data['labels']['all'][test_inds]
                train_batches, valid_batches, test_batches =\
                     all_data['batches']['all'][train_inds], all_data['batches']['all'][valid_inds], all_data['batches']['all'][test_inds]
                train_names, valid_names, test_names =\
                     all_data['names']['all'][train_inds], all_data['names']['all'][valid_inds], all_data['names']['all'][test_inds]
                # remove labels that are not in train
                unique_train_labels = np.unique(train_labels)
                valid_to_keep = np.array(
                    [i for i, l in enumerate(valid_labels) if l in unique_train_labels ]
                )
                test_to_keep = np.array(
                    [i for i, l in enumerate(test_labels) if l in unique_train_labels]
                )
                valid_data, valid_labels, valid_batches, valid_names =\
                      valid_data.iloc[valid_to_keep], valid_labels[valid_to_keep], valid_batches[valid_to_keep], valid_names[valid_to_keep]
                test_data, test_labels, test_batches, test_names =\
                      test_data.iloc[test_to_keep], test_labels[test_to_keep], test_batches[test_to_keep], test_names[test_to_keep]

                valid_inds, test_inds = valid_inds[valid_to_keep], test_inds[test_to_keep]

            elif self.args.train_on == 'all_lows':
                # keep all concs for train to be all_data['concs']['all'] == 'l'        
                train_inds = np.argwhere(all_data['concs']['all'] == 'l').flatten()
                valid_inds = np.argwhere(all_data['concs']['all'] == 'h').flatten()
                blanc_inds = np.argwhere(all_data['concs']['all'] == 'na').flatten()
                train_data, valid_data, blanc_data =\
                      all_data['inputs']['all'].iloc[train_inds], all_data['inputs']['all'].iloc[valid_inds], all_data['inputs']['all'].iloc[blanc_inds]
                train_labels, valid_labels, blanc_labels =\
                      all_data['labels']['all'][train_inds], all_data['labels']['all'][valid_inds], all_data['labels']['all'][blanc_inds]
                train_batches, valid_batches, blanc_batches =\
                      all_data['batches']['all'][train_inds], all_data['batches']['all'][valid_inds], all_data['batches']['all'][blanc_inds]
                train_names, valid_names, blanc_names =\
                      all_data['names']['all'][train_inds], all_data['names']['all'][valid_inds], all_data['names']['all'][blanc_inds]
                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
                blanc_nums = np.arange(0, len(blanc_labels))
                splitter = skf.split(blanc_nums, blanc_labels, blanc_batches)
                blanc_train_inds, blanc_valid_inds = splitter.__next__()
                train_data, valid_data = pd.concat((train_data, blanc_data.iloc[blanc_train_inds])), \
                    pd.concat((valid_data, blanc_data.iloc[blanc_valid_inds]))
                train_labels, valid_labels = np.concatenate((train_labels, blanc_labels[blanc_train_inds])), \
                    np.concatenate((valid_labels, blanc_labels[blanc_valid_inds]))
                train_batches, valid_batches = np.concatenate((train_batches, blanc_batches[blanc_train_inds])), \
                    np.concatenate((valid_batches, blanc_batches[blanc_valid_inds]))
                train_names, valid_names = np.concatenate((train_names, blanc_names[blanc_train_inds])), \
                    np.concatenate((valid_names, blanc_names[blanc_valid_inds]))
                

                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
                valid_nums = np.arange(0, len(valid_labels))
                splitter = skf.split(valid_nums, valid_labels, valid_batches)
                test_inds, valid_inds = splitter.__next__()
                test_data, valid_data = valid_data.iloc[test_inds], valid_data.iloc[valid_inds]
                test_labels, valid_labels = valid_labels[test_inds], valid_labels[valid_inds]
                test_batches, valid_batches = valid_batches[test_inds], valid_batches[valid_inds]
                test_names, valid_names = valid_names[test_inds], valid_names[valid_inds]

            elif self.args.train_on == 'all_highs':
                # keep all concs for train to be all_data['concs']['all'] == 'h'        
                train_inds = np.argwhere(all_data['concs']['all'] == 'h').flatten()
                valid_inds = np.argwhere(all_data['concs']['all'] == 'l').flatten()
                blanc_inds = np.argwhere(all_data['concs']['all'] == 'na').flatten()
                train_data, valid_data, blanc_data =\
                      all_data['inputs']['all'].iloc[train_inds], all_data['inputs']['all'].iloc[valid_inds], all_data['inputs']['all'].iloc[blanc_inds]
                train_labels, valid_labels, blanc_labels =\
                      all_data['labels']['all'][train_inds], all_data['labels']['all'][valid_inds], all_data['labels']['all'][blanc_inds]
                train_batches, valid_batches, blanc_batches =\
                      all_data['batches']['all'][train_inds], all_data['batches']['all'][valid_inds], all_data['batches']['all'][blanc_inds]
                train_names, valid_names, blanc_names =\
                      all_data['names']['all'][train_inds], all_data['names']['all'][valid_inds], all_data['names']['all'][blanc_inds]

                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
                blanc_nums = np.arange(0, len(blanc_labels))
                splitter = skf.split(blanc_nums, blanc_batches, blanc_batches)
                blanc_train_inds, blanc_valid_inds = splitter.__next__()
                train_data, valid_data = pd.concat((train_data, blanc_data.iloc[blanc_train_inds])), \
                    pd.concat((valid_data, blanc_data.iloc[blanc_valid_inds]))
                train_labels, valid_labels = np.concatenate((train_labels, blanc_labels[blanc_train_inds])), \
                    np.concatenate((valid_labels, blanc_labels[blanc_valid_inds]))
                train_batches, valid_batches = np.concatenate((train_batches, blanc_batches[blanc_train_inds])), \
                    np.concatenate((valid_batches, blanc_batches[blanc_valid_inds]))
                train_names, valid_names = np.concatenate((train_names, blanc_names[blanc_train_inds])), \
                    np.concatenate((valid_names, blanc_names[blanc_valid_inds]))

                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
                valid_nums = np.arange(0, len(valid_labels))
                splitter = skf.split(valid_nums, valid_batches, valid_batches)
                test_inds, valid_inds = splitter.__next__()
                test_data, valid_data = valid_data.iloc[test_inds], valid_data.iloc[valid_inds]
                test_labels, valid_labels = valid_labels[test_inds], valid_labels[valid_inds]
                test_batches, valid_batches = valid_batches[test_inds], valid_batches[valid_inds]
                test_names, valid_names = valid_names[test_inds], valid_names[valid_inds]

            lists['inds']['train'] += [train_inds]
            lists['inds']['valid'] += [valid_inds]
            lists['inds']['test'] += [test_inds]

            lists['names']['train'] += [all_data['names']['all'][train_inds]]
            lists['names']['valid'] += [all_data['names']['all'][valid_inds]]
            lists['names']['test'] += [all_data['names']['all'][test_inds]]

            lists['batches']['train'] += [train_batches]
            lists['batches']['valid'] += [valid_batches]
            lists['batches']['test'] += [test_batches]
            lists['unique_batches']['train'] += [list(np.unique(train_batches))]
            lists['unique_batches']['valid'] += [list(np.unique(valid_batches))]
            lists['unique_batches']['test'] += [list(np.unique(test_batches))]
            # Remove and add b10 from urinespositives to train and add to list
            if all_data['inputs']['urinespositives'].shape[0] > 0:
                b10_inds = np.argwhere(all_data['batches']['urinespositives'] == 'b10').flatten()
                train_data = pd.concat((train_data, all_data['inputs']['urinespositives'].iloc[b10_inds]))
                train_labels = np.concatenate((train_labels, all_data['labels']['urinespositives'][b10_inds]))
                train_batches = np.concatenate((train_batches, all_data['batches']['urinespositives'][b10_inds]))
                train_names = np.concatenate((train_names, all_data['names']['urinespositives'][b10_inds]))
                lists['names']['train'][-1] = np.concatenate((lists['names']['train'][-1], all_data['names']['urinespositives'][b10_inds]))
                lists['batches']['train'][-1] = np.concatenate((lists['batches']['train'][-1], all_data['batches']['urinespositives'][b10_inds]))
                lists['inds']['train'][-1] = np.concatenate((train_data.shape[0]))
                # Remove b10 from all_data
                all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'].drop(b10_inds)
                all_data['labels']['urinespositives'] = np.delete(all_data['labels']['urinespositives'], b10_inds)
                all_data['batches']['urinespositives'] = np.delete(all_data['batches']['urinespositives'], b10_inds)
                all_data['names']['urinespositives'] = np.delete(all_data['names']['urinespositives'], b10_inds)

            if n_aug > 0:
                train_data = augment_data(train_data, n_aug, p, g)
                train_data = np.nan_to_num(train_data)
                train_labels = np.concatenate([train_labels] * (n_aug + 1))
                train_batches = np.concatenate([train_batches] * (n_aug + 1))
            else:
                train_data = train_data.fillna(0)
                train_data = train_data.values
            valid_data = valid_data.fillna(0)
            test_data = test_data.fillna(0)

            lists['classes']['train'] += [
                np.array([np.argwhere(label == self.unique_labels)[0][0] for label in train_labels])
            ]
            lists['classes']['valid'] += [
                np.array([np.argwhere(label == self.unique_labels)[0][0] for label in valid_labels])
            ]
            lists['classes']['test'] += [
                np.array([np.argwhere(label == self.unique_labels)[0][0] for label in test_labels])
            ]
            lists['labels']['train'] += [train_labels]
            lists['labels']['valid'] += [valid_labels]
            lists['labels']['test'] += [test_labels]

            if self.args.model_name == 'xgboost' and 'cuda' in self.args.device:
                gpu_id = int(self.args.device.split(':')[-1])
                m = self.model(
                    device=f'cuda:{gpu_id}'
                )
            elif self.args.model_name == 'xgboost' and 'cuda' not in self.args.device:
                m = self.model()
            else:
                m = self.model()
            m.set_params(**param_grid)
            if self.args.ovr:
                m = OneVsRestClassifier(m)

            eval_set = [(valid_data.values, lists['classes']['valid'][-1])]
            m.fit(train_data, lists['classes']['train'][-1], eval_set=eval_set, verbose=True)

            self.dump_model(h, m, scaler_name, lists)
            best_iteration += [m.best_iteration]
            try:
                lists['acc']['train'] += [m.score(train_data, lists['classes']['train'][-1])]
                lists['preds']['train'] += [m.predict(train_data)]
            except:
                lists['acc']['train'] += [m.score(train_data.values, lists['classes']['train'][-1])]
                lists['preds']['train'] += [m.predict(train_data.values)]

            try:
                lists['acc']['valid'] += [m.score(valid_data, lists['classes']['valid'][-1])]
                lists['acc']['test'] += [m.score(test_data, lists['classes']['test'][-1])]
                lists['preds']['valid'] += [m.predict(valid_data)]
                lists['preds']['test'] += [m.predict(test_data)]
            except:
                lists['acc']['valid'] += [m.score(valid_data.values, lists['classes']['valid'][-1])]
                lists['acc']['test'] += [m.score(test_data.values, lists['classes']['test'][-1])]
                lists['preds']['valid'] += [m.predict(valid_data.values)]
                lists['preds']['test'] += [m.predict(test_data.values)]

            if all_data['inputs']['urinespositives'].shape[0] > 0:
                try:
                    lists['preds']['posurines'] += [m.predict(all_data['inputs']['urinespositives'])]
                    try:
                        lists['proba']['posurines'] += [m.predict_proba(all_data['inputs']['urinespositives'])]
                    except:
                        pass

                except:
                    lists['preds']['posurines'] += [m.predict(all_data['inputs']['urinespositives'].values)]
                    try:
                        lists['proba']['posurines'] += [m.predict_proba(all_data['inputs']['urinespositives'].values)]
                    except:
                        pass

            try:
                lists['proba']['train'] += [m.predict_proba(train_data)]
                lists['proba']['valid'] += [m.predict_proba(valid_data)]
                lists['proba']['test'] += [m.predict_proba(test_data)]
                data_list = {
                    'valid': {
                        'inputs': valid_data,
                        'labels': lists['classes']['valid'][-1],
                        'preds': lists['preds']['valid'][-1],
                        'proba': lists['proba']['valid'][-1],
                        'batches': valid_batches,
                        'names': all_data['names']['valid']
                    },
                    'test': {
                        'inputs': test_data,
                        'labels': lists['classes']['test'][-1],
                        'preds': lists['preds']['test'][-1],
                        'proba': lists['proba']['test'][-1],
                        'batches': test_batches,
                        'names': all_data['names']['test']
                    }
                }
            except:
                data_list = {
                    'valid': {
                        'inputs': valid_data,
                        'labels': lists['classes']['valid'][-1],
                        'preds': lists['preds']['valid'][-1],
                        'batches': valid_batches,
                        'names': all_data['names']['valid']
                    },
                    'test': {
                        'inputs': test_data,
                        'labels': lists['classes']['test'][-1],
                        'preds': lists['preds']['test'][-1],
                        'batches': test_batches,
                        'names': all_data['names']['test']
                    }
                }
            lists['mcc']['train'] += [MCC(lists['classes']['train'][-1], lists['preds']['train'][-1])]
            lists['mcc']['valid'] += [MCC(lists['classes']['valid'][-1], lists['preds']['valid'][-1])]
            lists['mcc']['test'] += [MCC(lists['classes']['test'][-1], lists['preds']['test'][-1])]
            # Initialize best_scores on first iteration
            if self.best_scores['mcc']['valid'] is None:
                self.retrieve_best_scores(lists)

            h += 1

        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        log_ord(self.data, self.uniques, ord_path, scaler_name, run)
        data = copy.deepcopy(self.data)
        metrics = log_fct(data, scaler_name, metrics)
        log_ord(data, self.uniques2, ord_path, f'{scaler_name}_blancs', run)

        if self.args.groupkfold:
            batches = np.concatenate([np.unique(x) for x in lists['batches']['valid']])
            toprint = [f"{batches[i]}:{lists['mcc']['valid'][i]}" for i in range(len(batches))]
            print(toprint)
        else:
            print(lists['mcc']['valid'])
        print('valid_acc:', np.mean(lists['acc']['valid']), \
              'valid_mcc:', np.mean(lists['mcc']['valid']), \
              'scaler:', scaler_name,
              'h_params:', param_grid
              )
        posurines_df = None  # TODO move somewhere more logical

        # Log in neptune the optimal iteration
        if self.args.log_neptune:
            model["best_iteration"] = run["best_iteration"] = np.round(np.mean([x for x in best_iteration]))
            model["model_size"] = run["model_size"] = get_size_in_mb(m)

        lists, posurines_df = self.save_confusion_matrices(all_data, lists, run)
        current_valid_mcc_mean = float(np.mean(lists['mcc']['valid'])) if len(lists['mcc']['valid']) > 0 else -np.inf
        prev_valid_mcc_mean = (
            float(np.mean(self.best_scores['mcc']['valid']))
            if isinstance(self.best_scores['mcc']['valid'], list) and len(self.best_scores['mcc']['valid']) > 0
            else (-np.inf if self.best_scores['mcc']['valid'] is None else float(self.best_scores['mcc']['valid']))
        )
        if current_valid_mcc_mean > prev_valid_mcc_mean:
            self.save_roc_curves(lists, run)
            if self.args.log_shap:
                print('log shap')
                if self.args.model_name == 'xgboost' and 'cuda' in self.args.device:
                    gpu_id = int(self.args.device.split(':')[-1])
                    m = self.model(
                        device=f'cuda:{gpu_id}'
                    )
                elif self.args.model_name == 'xgboost' and 'cuda' not in self.args.device:
                    m = self.model()
                else:
                    m = self.model()
                # Remove early_stopping_rounds from param_grid
                param_grid.pop('early_stopping_rounds', None)
                param_grid['n_estimators'] = best_iteration[-1]

                m.set_params(**param_grid)
                if self.args.ovr:
                    m = OneVsRestClassifier(m)

                m.fit(train_data, lists['classes']['train'][-1], verbose=True)
                Xs = {
                    'train': train_data,
                    'valid': valid_data,
                    'test': test_data,
                    # 'posurines': all_data['inputs']['urinespositives'],
                }
                ys = {
                    'train': lists['classes']['train'][-1],
                    'valid': lists['classes']['valid'][-1],
                    'test': lists['classes']['test'][-1],
                    # 'posurines': np.array([
                    # np.argwhere(l == self.unique_labels)[0][0] for l in all_data['labels']['urinespositives']]),
                }
                args_dict = {
                    'inputs': Xs,
                    'labels': ys,
                    'model': m,
                    'model_name': self.args.model_name,
                    'log_path': f'logs/{self.args.model_name}',
                }
                run = log_shap(run, args_dict)
            # save the features kept
            with open(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl', 'wb') as f:
                pickle.dump(data_list['test']['inputs'].columns, f)

            self.retrieve_best_scores(lists)
            if self.keep_models(scaler_name):
                # Save the individual scores of each sample with class, #batch
                self.save_results_df(lists, run)
                best_scores = self.save_best_model_hparams(
                    param_grid, other_params, scaler_name, lists['unique_batches'], metrics
                )
        else:
            best_scores = {
                'nbe': None,
                'ari': None,
                'ami': None,
            }
            self.remove_models(scaler_name)

        if all_data['inputs']['urinespositives'].shape[0] > 0 and posurines_df is not None:
            self.save_thresholds_curve0('posurines', posurines_df, run)
            self.save_thresholds_curve('posurines', lists, run)
            run[f'posurines/individual_results'].upload(
                f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
            )

        if self.args.log_neptune:
            log_neptune(run, lists, best_scores)
        # TODO THIS SHOULD BE in utils or something
        if self.args.log_dvclive:
            self.live.next_step()
            # try:
            #     # aggregate current best metrics
            #     to_log = {}
            #     for m in ['acc', 'mcc']:
            #         if m in lists and isinstance(lists[m]['valid'], list) and len(lists[m]['valid']) > 0:
            #             to_log[f'valid/{m}'] = float(np.mean(lists[m]['valid']))
            #     if m in lists and isinstance(lists[m]['test'], list) and len(lists[m]['test']) > 0:
            #         to_log[f'test/{m}'] = float(np.mean(lists[m]['test']))
            # best_scores extras
            # for k in ['ari', 'ami', 'nbe']:
            #     if k in best_scores:
            #         for g in ['train', 'valid', 'test']:
            #             if g in best_scores[k]:
            #                 to_log[f'{g}/{k}'] = best_scores[k][g]
            # for k,v in to_log.items():
            #     self.live.log_metric(k, v)
            #     self.live.next_step()
            # except Exception as e:
            #     print('dvclive log failed', e)
        if self.args.log_neptune:
            run.stop()
            model.stop()

        return 1 - np.mean(lists['mcc']['valid'])

    def save_best_model_hparams(self, params, other_params, scaler_name, unique_batches, metrics):
        param_grid = {}

        for name in params:
            param_grid[name] = params[name]
            if isinstance(params[name], np.int64):
                param_grid[name] = int(params[name])
            if isinstance(params[name], np.bool_):
                param_grid[name] = int(params[name])
        for name in other_params:
            param_grid[name] = other_params[name]
            if isinstance(other_params[name], np.int64):
                param_grid[name] = int(other_params[name])
        param_grid['scaler'] = scaler_name
        self.best_params_dict = param_grid

        # Convert the data before saving
        serializable_batches = convert_to_serializable(unique_batches)

        # Now save to JSON
        with open(
            f'{self.log_path}/saved_models/unique_batches_{self.name}_{self.args.model_name}.json',
            "w",
        ) as read_file:
            json.dump(serializable_batches, read_file)
        self.best_params_dict_values['train_batches'] = self.best_scores['acc']['train']

        self.best_params_dict_values['train_acc'] = self.best_scores['acc']['train']
        self.best_params_dict_values['valid_acc'] = self.best_scores['acc']['valid']
        self.best_params_dict_values['test_acc'] = self.best_scores['acc']['test']
        self.best_params_dict_values['train_acc'] = self.best_scores['acc']['train']
        self.best_params_dict_values['valid_acc'] = self.best_scores['acc']['valid']
        self.best_params_dict_values['test_acc'] = self.best_scores['acc']['test']

        self.best_params_dict_values['train_mcc'] = self.best_scores['mcc']['train']
        self.best_params_dict_values['valid_mcc'] = self.best_scores['mcc']['valid']
        self.best_params_dict_values['test_mcc'] = self.best_scores['mcc']['test']
        self.best_params_dict_values['train_mcc'] = self.best_scores['mcc']['train']
        self.best_params_dict_values['valid_mcc'] = self.best_scores['mcc']['valid']
        self.best_params_dict_values['test_mcc'] = self.best_scores['mcc']['test']

        n_batches = np.unique(
            np.concatenate([
                np.concatenate(unique_batches[b]) for b in unique_batches if len(unique_batches[b]) > 0
            ])
        ).flatten().shape[0]
        for name in metrics.keys():
            self.best_params_dict[f'{name}/ami'] = metrics[name]['all']['adjusted_mutual_info_score']['domains']
            self.best_params_dict[f'{name}/ari'] = metrics[name]['all']['adjusted_rand_score']['domains']
            self.best_params_dict[f'{name}/nBE'] = (
                np.log(n_batches) - metrics[name]['all']['shannon']['domains']
            ) / np.log(n_batches)

        self.best_params_dict['train_acc_mean'] = np.mean(self.best_scores['acc']['train'])
        self.best_params_dict['valid_acc_mean'] = np.mean(self.best_scores['acc']['valid'])
        self.best_params_dict['test_acc_mean'] = np.mean(self.best_scores['acc']['test'])
        self.best_params_dict['train_acc_std'] = np.std(self.best_scores['acc']['train'])
        self.best_params_dict['valid_acc_std'] = np.std(self.best_scores['acc']['valid'])
        self.best_params_dict['test_acc_std'] = np.std(self.best_scores['acc']['test'])

        self.best_params_dict['train_mcc_mean'] = np.mean(self.best_scores['mcc']['train'])
        self.best_params_dict['valid_mcc_mean'] = np.mean(self.best_scores['mcc']['valid'])
        self.best_params_dict['test_mcc_mean'] = np.mean(self.best_scores['mcc']['test'])
        self.best_params_dict['train_mcc_std'] = np.std(self.best_scores['mcc']['train'])
        self.best_params_dict['valid_mcc_std'] = np.std(self.best_scores['mcc']['valid'])
        self.best_params_dict['test_mcc_std'] = np.std(self.best_scores['mcc']['test'])

        with open(
            f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}.json',
            "w",
        ) as read_file:
            json.dump(self.best_params_dict, read_file)
        with open(
            f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}_values.json',
            "w",
        ) as read_file:
            json.dump(self.best_params_dict_values, read_file)
        # add ari, ami and nbe to run
        return self.best_params_dict

    def load_best_model_hparams(self, load_models: bool = False, load_scaler: bool = True):
        """
        Load previously saved best params / values / unique batches (and optionally the model(s) & scaler).

        Returns:
            dict with keys:
              params              -> dict of hyperparameters + aggregated metrics
              values              -> raw per-split score lists (if available)
              unique_batches      -> structure saved at training
              scaler_name         -> scaler name stored in params (if present)
              models              -> list of loaded model objects (if load_models=True)
              scaler              -> scaler object (if load_scaler and available)
        """
        saved_dir = f'{self.log_path}/saved_models'
        best_params_path = f'{saved_dir}/best_params_{self.name}_{self.args.model_name}.json'
        best_values_path = f'{saved_dir}/best_params_{self.name}_{self.args.model_name}_values.json'
        unique_batches_path = f'{saved_dir}/unique_batches_{self.name}_{self.args.model_name}.json'

        def _maybe_dvc_pull(target):
            # Attempt a silent dvc pull if metafile exists but target missing
            if not os.path.exists(target) and os.path.exists(target + '.dvc'):
                try:
                    os.system(f'dvc pull "{target}" >/dev/null 2>&1')
                except Exception:
                    pass

        for p in [best_params_path, best_values_path, unique_batches_path]:
            _maybe_dvc_pull(p)

        if not os.path.exists(best_params_path):
            raise FileNotFoundError(f'Best params file not found: {best_params_path}')

        with open(best_params_path, 'r') as rf:
            self.best_params_dict = json.load(rf)
        if os.path.exists(best_values_path):
            with open(best_values_path, 'r') as rf:
                self.best_params_dict_values = json.load(rf)
        else:
            self.best_params_dict_values = {}

        if os.path.exists(unique_batches_path):
            with open(unique_batches_path, 'r') as rf:
                unique_batches = json.load(rf)
        else:
            unique_batches = {}

        scaler_name = self.best_params_dict.get('scaler', None)

        loaded_models = []
        scaler_obj = None

        mode_tag = getattr(self.args, 'mode', 'nomode')

        if load_models:
            # Collect all promoted (non _tmp) model files for this scaler / mode
            pattern_prefix = f'{self.args.model_name}_{mode_tag}_{scaler_name}_'
            for fname in sorted(os.listdir(saved_dir)):
                if fname.startswith(pattern_prefix) and fname.endswith('.pkl') and '_indices_' not in fname \
                        and '_meta' not in fname and not fname.endswith('_tmp.pkl'):
                    full_path = os.path.join(saved_dir, fname)
                    _maybe_dvc_pull(full_path)
                    try:
                        with open(full_path, 'rb') as mf:
                            loaded_models.append(pickle.load(mf))
                    except Exception as e:
                        print(f'Warning: failed to load model {fname}: {e}')

        if load_scaler and scaler_name is not None:
            scaler_path = f'{saved_dir}/scaler_{scaler_name}.pkl'
            _maybe_dvc_pull(scaler_path)
            if os.path.exists(scaler_path):
                try:
                    with open(scaler_path, 'rb') as sf:
                        scaler_obj = pickle.load(sf)
                except Exception as e:
                    print(f'Warning: failed to load scaler {scaler_name}: {e}')

        # Restore best_scores minimal (means) if lists not present
        try:
            for metric in ['acc', 'mcc']:
                for split in ['train', 'valid', 'test']:
                    mean_key = f'{split}_{metric}_mean'
                    if mean_key in self.best_params_dict:
                        # store as single-element list for compatibility with downstream code
                        self.best_scores[metric][split] = [self.best_params_dict[mean_key]]
        except Exception:
            pass

        return {
            'params': self.best_params_dict,
            'values': self.best_params_dict_values,
            'unique_batches': unique_batches,
            'scaler_name': scaler_name,
            'models': loaded_models,
            'scaler': scaler_obj,
        }

    def make_predictions(self, lists, run):
        blanc_ids = [
            np.array([i for i, (x, b) in enumerate(
                zip(lists['labels']['test'][j], lists['batches']['test'][j])
            ) if x == 'blanc' and b in [np.argwhere(
                self.uniques['batches'] == 'b11'
            )[0][0]]]) for j in range(len(lists['labels']['test']))
        ]
        try:
            np.concatenate(lists['proba']['posurines']).max(1)
        except Exception as e:
            print(f'DIM PROBLEM: {e}')
        posurines_df = pd.DataFrame(
            {
                'names': np.concatenate(lists['names']['posurines']),
                'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['posurines'])]),
                'batches': np.concatenate(lists['batches']['posurines']),
                'preds': np.concatenate(lists['preds']['posurines']),
                'labels': np.concatenate(lists['labels']['posurines']),
                'proba': np.concatenate(lists['proba']['posurines']).max(1),
            }
        )
        for i, label in enumerate(self.unique_labels):
            posurines_df[label] = np.concatenate(lists['proba']['posurines'])[:, i]

        posurines_df.loc[:, 'preds'] = [
            self.unique_labels[int(label)] for label in posurines_df.loc[:, 'preds'].to_numpy()
        ]
        # assert 'bact' not in posurines_df.loc[:, 'preds']
        posurines_df.to_csv(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
        )
        try:
            lists['proba']['posurines'] = np.stack(lists['proba']['posurines'])
        except Exception as e:
            print(f'PROBLEM WITH PROBA: {e}')
            return posurines_df, lists

        def flatten(xss):
            return [x for xs in xss for x in xs]

        blk_names = flatten([
            [lists['names']['test'][j][i] for i in blanc_ids[j]]
            for j in range(len(lists['names']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        blk_batches = flatten([
            [lists['batches']['test'][j][i] for i in blanc_ids[j]]
            for j in range(len(lists['batches']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        blk_labels = flatten([
            [lists['labels']['test'][j][i] for i in blanc_ids[j]]
            for j in range(len(lists['labels']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        blk_preds = flatten([
            [lists['preds']['test'][j][i] for i in blanc_ids[j]]
            for j in range(len(lists['preds']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        proba_posurines = np.stack(lists['proba']['posurines']).mean(0)
        blk_proba = np.stack(flatten([
            [lists['proba']['test'][j][i] for i in blanc_ids[j]] for j in range(
                len(lists['preds']['posurines'])
                ) if len(blanc_ids[j]) > 0
        ]))
        if self.args.groupkfold:
            # Find which or the batch indices are b10 and bpatients
            # b10_ind = [
            # i for i, b in enumerate(lists['batches']['train']) if np.argwhere(
            # self.uniques['batches'] == 'b10')[0][0] in b]
            reps_to_use = [0 if np.argwhere(self.uniques['batches'] == 'b11')[0][0] in lists['batches']['train'][i]
                           or (np.argwhere(self.uniques['batches'] == 'b10')[0][0] not in lists['batches']['train'][i]
                               or np.argwhere(self.uniques['batches'] == 'bpatients')[0][0]
                               not in lists['batches']['train'][i]
                               ) else 1 for i in range(
                                len(lists['proba']['posurines']))]
            # use only the reps of reps_to_use in the proba

            if sum(reps_to_use) > 0:
                lists['proba']['posurines'] = np.stack([
                    lists['proba']['posurines'][i] for i in range(
                        len(lists['proba']['posurines'])
                    ) if reps_to_use[i] == 1
                ])
            else:
                print('No upos in test!')
                print(f'test: {self.args.test}')
                posurines_df = pd.DataFrame([])
                return posurines_df, lists

        posurines_df = pd.DataFrame(
            {
                'names': lists['names']['posurines'][0].tolist() + blk_names,
                'batches': lists['batches']['posurines'][0].tolist() + blk_batches,
                'preds': lists['preds']['posurines'][0].tolist() + blk_preds,
                'labels': lists['labels']['posurines'][0].tolist() + blk_labels,
                'proba': np.concatenate((proba_posurines.max(1).flatten(), blk_proba.max(1).flatten())),
            }
        )
        proba_posurines = np.concatenate((proba_posurines, blk_proba))
        for i, label in enumerate(self.unique_labels):
            posurines_df[label] = proba_posurines[:, i]

        posurines_df.loc[:, 'preds'] = [
            self.unique_labels[int(label)] for label in posurines_df.loc[:, 'preds'].to_numpy()
        ]
        assert 'bact' not in posurines_df.loc[:, 'preds']
        posurines_df.to_csv(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_mode_results.csv'
        )
        run['posurines/individual_results'].upload(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
        )
        run['posurines/mode_results'].upload(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_mode_results.csv'
        )

        return posurines_df, lists

    def make_predictions_old(self, lists, run):
        blanc_ids = [
            np.array([i for i, (x, b) in enumerate(
                zip(lists['labels']['valid'][j], lists['batches']['valid'][j])
            ) if x == 'blanc' and b in [np.argwhere(
                self.uniques['batches'] == 'b11'
            )[0][0]]]) for j in range(len(lists['labels']['valid']))
        ]
        try:
            np.concatenate(lists['proba']['posurines']).max(1)
        except Exception as e:
            print(f"Error in make_predictions: {e}")
            print('DIM PROBLEM')
        posurines_df = pd.DataFrame(
            {
                'names': np.concatenate(lists['names']['posurines']),
                'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['posurines'])]),
                'batches': np.concatenate(lists['batches']['posurines']),
                'preds': np.concatenate(lists['preds']['posurines']),
                'labels': np.concatenate(lists['labels']['posurines']),
                'proba': np.concatenate(lists['proba']['posurines']).max(1),
            }
        )
        for i, label in enumerate(self.unique_labels):
            posurines_df[label] = np.concatenate(lists['proba']['posurines'])[:, i]

        posurines_df.loc[:, 'preds'] = [
            self.unique_labels[int(label)] for label in posurines_df.loc[:, 'preds'].to_numpy()
        ]
        # assert 'bact' not in posurines_df.loc[:, 'preds']
        posurines_df.to_csv(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
        )
        try:
            lists['proba']['posurines'] = np.stack(lists['proba']['posurines'])
        except Exception as e:
            print(f"Error in make_predictions: {e}")
            return posurines_df, lists

        def flatten(xss):
            return [x for xs in xss for x in xs]
        blk_names = flatten([
            [
                lists['names']['valid'][j][i] for i in blanc_ids[j]
            ] for j in range(len(lists['names']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        blk_batches = flatten([
            [
                lists['batches']['valid'][j][i] for i in blanc_ids[j]
            ] for j in range(len(lists['batches']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        blk_labels = flatten([
            [
                lists['labels']['valid'][j][i] for i in blanc_ids[j]
            ] for j in range(len(lists['labels']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        blk_preds = flatten([
            [
                lists['preds']['valid'][j][i] for i in blanc_ids[j]
            ] for j in range(len(lists['preds']['posurines'])) if len(blanc_ids[j]) > 0
        ])
        if self.args.groupkfold:
            reps_to_use = [0 if np.argwhere(self.uniques['batches'] == 'b11')[0][0] in lists['batches']['train'][i] or (
                np.argwhere(self.uniques['batches'] == 'b10')[0][0] not in lists['batches']['train'][i] or
                np.argwhere(self.uniques['batches'] == 'bpatients')[0][0] not in lists['batches']['train'][i]
            ) else 1 for i in range(len(lists['proba']['posurines']))]
            # use only the reps of reps_to_use in the proba
            lists['proba']['posurines'] = np.stack([
                lists['proba']['posurines'][i] for i in range(len(lists['proba']['posurines'])) if reps_to_use[i] == 1
            ])

        proba_posurines = np.stack(lists['proba']['posurines']).mean(0)
        blk_proba = np.stack(flatten([
            [lists['proba']['valid'][j][i] for i in blanc_ids[j]] for j in range(
                len(lists['preds']['posurines'])
                ) if len(blanc_ids[j]) > 0
        ]))

        posurines_df = pd.DataFrame(
            {
                'names': lists['names']['posurines'][0].tolist() + blk_names,
                'batches': lists['batches']['posurines'][0].tolist() + blk_batches,
                'preds': lists['preds']['posurines'][0].tolist() + blk_preds,
                'labels': lists['labels']['posurines'][0].tolist() + blk_labels,
                'proba': np.concatenate((proba_posurines.max(1).flatten(), blk_proba.max(1).flatten())),
            }
        )
        proba_posurines = np.concatenate((proba_posurines, blk_proba))
        for i, label in enumerate(self.unique_labels):
            posurines_df[label] = proba_posurines[:, i]

        posurines_df.loc[:, 'preds'] = [
            self.unique_labels[int(label)] for label in posurines_df.loc[:, 'preds'].to_numpy()
        ]
        assert 'bact' not in posurines_df.loc[:, 'preds']
        posurines_df.to_csv(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_mode_results.csv'
        )
        run['posurines/individual_results'].upload(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
        )
        run['posurines/mode_results'].upload(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_mode_results.csv'
        )

        return posurines_df, lists

    def save_results_df(self, lists, run):
        if len(lists['proba']['train']) > 0:
            df_valid = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['valid']),
                    'labels': np.concatenate(lists['labels']['valid']),
                    'batches': np.concatenate(lists['batches']['valid']),
                    'preds': np.concatenate(lists['preds']['valid']),
                    'proba': np.concatenate(lists['proba']['valid']).max(1),
                    'names': np.concatenate(lists['names']['valid']),
                    'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['valid'])])
                }
            )
            df_test = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['test']),
                    'labels': np.concatenate(lists['labels']['test']),
                    'batches': np.concatenate(lists['batches']['test']),
                    'preds': np.concatenate(lists['preds']['test']),
                    'proba': np.concatenate(lists['proba']['test']).max(1),
                    'names': np.concatenate(lists['names']['test']),
                    'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['test'])])
                }
            )
            for i, label in enumerate(self.unique_labels):
                df_valid[label] = np.concatenate(lists['proba']['valid'])[:, i]
                df_test[label] = np.concatenate(lists['proba']['test'])[:, i]

        else:
            df_valid = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['valid']),
                    'labels': np.concatenate(lists['labels']['valid']),
                    'batches': np.concatenate(lists['batches']['valid']),
                    'preds': np.concatenate(lists['preds']['valid']),
                    'names': np.concatenate(lists['names']['valid']),
                    'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['valid'])])
                }
            )
            df_test = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['test']),
                    'labels': np.concatenate(lists['labels']['test']),
                    'batches': np.concatenate(lists['batches']['test']),
                    'preds': np.concatenate(lists['preds']['test']),
                    'names': np.concatenate(lists['names']['test']),
                    'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['test'])])
                }
            )
        df_valid.loc[:, 'preds'] = [
            self.unique_labels[int(label)] for label in df_valid.loc[:, 'preds'].to_numpy()
        ]
        df_test.loc[:, 'preds'] = [
            self.unique_labels[int(label)] for label in df_test.loc[:, 'preds'].to_numpy()
        ]

        # Provide batches_labels column for downstream logging consistency
        try:
            if 'batches_labels' not in df_valid.columns:
                # Map integer batch indices back to label names if uniques available
                if hasattr(self, 'uniques') and 'batches_labels' in self.uniques:
                    batch_map = {i: b for i, b in enumerate(self.uniques['batches_labels'])}
                    df_valid['batches_labels'] = [batch_map.get(int(b), b) for b in df_valid['batches']]
                    df_test['batches_labels'] = [batch_map.get(int(b), b) for b in df_test['batches']]
                elif hasattr(self, 'uniques') and 'batches' in self.uniques:
                    batch_map = {i: b for i, b in enumerate(self.uniques['batches'])}
                    df_valid['batches_labels'] = [batch_map.get(int(b), b) for b in df_valid['batches']]
                    df_test['batches_labels'] = [batch_map.get(int(b), b) for b in df_test['batches']]
                else:
                    # Fallback: copy existing batches column
                    df_valid['batches_labels'] = df_valid['batches']
                    df_test['batches_labels'] = df_test['batches']
        except Exception:
            # On any failure, fallback silently to copying batches
            if 'batches_labels' not in df_valid.columns:
                df_valid['batches_labels'] = df_valid.get('batches', [])
            if 'batches_labels' not in df_test.columns:
                df_test['batches_labels'] = df_test.get('batches', [])

        df_valid.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_valid_individual_results.csv')
        df_test.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_test_individual_results.csv')
        run['valid/individual_results'].upload(
            f'{self.log_path}/saved_models/{self.args.model_name}_valid_individual_results.csv'
        )
        run['test/individual_results'].upload(
            f'{self.log_path}/saved_models/{self.args.model_name}_test_individual_results.csv'
        )
        self.save_thresholds_curve('valid', lists, run)
        self.save_thresholds_curve('test', lists, run)
        self.save_thresholds_curve0('valid', df_valid, run)
        self.save_thresholds_curve0('test', df_test, run)
        # Do the same but with posurines, if any

        plot_bars(self.args, run, self.unique_labels)

    def save_calibration_curves(self, lists, run):
        # Use cross validation iterations stored in lists, loop over dont concat
        # import calibration_curve
        for group in ['train', 'valid', 'test', 'posurines']:
            try:
                for i in range(len(lists['proba'][group])):
                    fig = plt.figure()
                    proba = lists['proba'][group][i]
                    classes = lists['classes'][group][i]
                    # names = lists['names'][group][i]
                    for j, label in enumerate(self.unique_labels):
                        binary_class = [1 if c == j else 0 for c in classes]
                        fop, mpv = calibration_curve(binary_class, proba[:, j], n_bins=10)
                        plt.plot(mpv, fop, marker='o', label=label)
                    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
                    plt.legend()
                    plt.xlabel('TPR')
                    plt.ylabel('TNR')
                    fig.savefig(f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_calibration.png')
                    run[f'{group}/calibration'].upload(
                        f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_calibration.png'
                    )
            except Exception as e:
                print('Problem in save_calibration_curves', e)

    def save_calibration_intervals(self, lists, run):
        try:
            for group in ['train', 'valid', 'test', 'posurines']:
                fig, ax = plt.subplots()
                for j, label in enumerate(self.unique_labels):
                    # Collect all FOP and MPV from each cross-validation iteration
                    prob_true = []
                    prob_pred = []

                    for i in range(len(lists['proba'][group])):
                        proba = lists['proba'][group][i]
                        classes = lists['classes'][group][i]
                        binary_class = [1 if c == j else 0 for c in classes]

                        # Compute the calibration curve for this iteration
                        fop, mpv = calibration_curve(binary_class, proba[:, j], n_bins=10)
                        prob_true.append(fop)
                        prob_pred.append(mpv)

                _ = self.plot_calibration_intervals({'prob_true': prob_true, 'prob_pred': prob_pred}, ax)

                plt.xlabel('Mean Predicted Value')
                plt.ylabel("Fraction of Positives")
                os.makedirs("results/calibration/", exist_ok=True)
                plt.savefig(f'results/calibration/{self.name}_{self.args.model_name}_{group}_calibration_CI.png')
                plt.savefig(f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_calibration_CI.png')
                run[f'{group}/calibration_CI'].upload(
                    f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_calibration_CI.png'
                )
        except Exception as e:
            print(f"Error generating calibration curve for {group}: {e}")
            pass

    def plot_calibration_intervals(results, ax):
        # Compute mean and standard deviation for calibration curve
        fpr_mean = np.linspace(0, 1, 100)
        interp_tprs = []
        for i in range(len(results['prob_true'])):
            interp_tpr = np.interp(fpr_mean, results['prob_true'][i], results['prob_pred'][i])
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)

        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_std = np.std(interp_tprs, axis=0)

        ax.set_title("Calibration Curve")            
        tpr_upper = np.clip(tpr_mean + tpr_std, 0, 1)
        tpr_lower = tpr_mean - tpr_std
        ax.plot(fpr_mean, tpr_mean, lw=2, label="Mean Calibration Curve")
        ax.fill_between(fpr_mean, tpr_lower, tpr_upper, alpha=.2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='black')
        ax.legend(loc="lower right")

        return ax

    def save_roc_curves(self, lists, run):
        try:
            self.best_roc_train = plot_roc(lists['proba']['train'], lists['classes']['train'], self.unique_labels,
                                           f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_train",
                                           binary=self.args.binary, acc=lists['acc']['train'], run=run)
        except Exception as e:
            print('Problem in save_roc_curves train:', e)
        try:
            self.best_roc_valid = plot_roc(lists['proba']['valid'], lists['classes']['valid'], self.unique_labels,
                                           f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_valid",
                                           binary=self.args.binary, acc=lists['acc']['valid'], run=run)
        except Exception as e:
            print('Problem in save_roc_curves valid:', e)
        try:
            self.best_roc_test = plot_roc(lists['proba']['test'], lists['classes']['test'], self.unique_labels,
                                          f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_test",
                                          binary=self.args.binary, acc=lists['acc']['test'], run=run)
        except Exception as e:
            print('Problem in save_roc_curves test:', e)
        
        try:
            self.best_roc_posurines = plot_roc(
                lists['proba']['posurines'], lists['classes']['posurines'], self.unique_labels,
                f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_posurines",
                binary=self.args.binary, acc=lists['acc']['posurines'], run=run
            )
        except Exception as e:
            print('Problem in save_roc_curves posurines:', e)

    def save_thresholds_curve0(self, group, df, run):
        accs = []
        mccs = []
        proportion_predicted = []
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thres in thresholds:
            df1 = df.copy()
            inds = [i for i, proba in enumerate(df.loc[:, 'proba'].to_numpy()) if proba > thres]
            df1 = df1.iloc[inds]
            df1.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_{thres}.csv')
            run[f'{group}/individual_results_{thres}'].upload(
                f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_{thres}.csv'
            )
            accs += [ACC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
            mccs += [MCC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
            proportion_predicted += [df1.shape[0] / df.shape[0]]
        fig = plt.figure()
        plt.plot(thresholds, accs, label='acc')
        plt.plot(thresholds, mccs, label='mcc')
        plt.plot(thresholds, proportion_predicted, label='FOT')  # Fraction > threshold
        # Add a dash line at 0.95 on y
        plt.axhline(0.95, color='black', linestyle='--')
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        # plt.xlim(0, 9)
        fig.savefig(f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_thresholds.png')
        run[f'{group}/thresholds'].upload(
            f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_thresholds.png'
        )

    def save_thresholds_curve(self, group, lists, run):
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        accs = {x: [] for x in thresholds}
        mccs = {x: [] for x in thresholds}
        proportion_predicted = {x: [] for x in thresholds}
        for batch in range(len(lists['proba'][group])):
            try:
                probs = lists['proba'][group][batch].max(1)
            except Exception as e:
                print('Problem in save_thresholds_curve', e)
                probs = lists['proba'][group][batch]
            df = pd.DataFrame(
                {
                    'classes': lists['classes'][group][batch],
                    'labels': lists['labels'][group][batch],
                    'batches': lists['batches'][group][batch],
                    'preds': lists['preds'][group][batch],
                    'proba': probs,
                    'names': lists['names'][group][batch]
                }
            )

            df.loc[:, 'preds'] = [
                self.unique_labels[int(label)] for label in df.loc[:, 'preds'].to_numpy()
            ]

            if len(np.unique(df.loc[:, 'batches'].to_numpy())) == 1:
                batch_name = df.loc[:, 'batches'].to_numpy()[0]
            else:
                batch_name = batch
            for thres in thresholds:
                df1 = df.copy()
                inds = [i for i, proba in enumerate(df.loc[:, 'proba'].to_numpy()) if proba > thres]
                df1 = df1.iloc[inds]
                df1.to_csv(
                    f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_'
                    f'{thres}_{batch_name}.csv'
                )
                run[f'{group}/individual_results_{thres}_{batch_name}'].upload(
                    f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_'
                    f'{thres}_{batch_name}.csv'
                    )
                accs[thres] += [ACC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
                mccs[thres] += [MCC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
                proportion_predicted[thres] += [df1.shape[0] / df.shape[0]]
        fig = plt.figure()
        plt.axhline(0.95, color='black', linestyle='--')
        plt.plot(thresholds, [np.mean(accs[k]) for k in accs.keys()], label='acc', color='blue')
        plt.plot(thresholds, [np.mean(mccs[k]) for k in mccs.keys()], label='mcc', color='red')
        plt.plot(thresholds, [
            np.mean(proportion_predicted[k]) for k in proportion_predicted.keys()
        ], label='FOT', color='black')  # Fraction > threshold
        plt.errorbar(thresholds, [np.mean(accs[k]) for k in accs.keys()],
                     [np.std(accs[k]) for k in accs.keys()], fmt='', color='blue')
        plt.errorbar(thresholds, [np.mean(mccs[k]) for k in mccs.keys()],
                     [np.std(mccs[k]) for k in mccs.keys()], fmt='', color='red')
        plt.errorbar(thresholds, [np.mean(proportion_predicted[k]) for k in proportion_predicted.keys()],
                     [np.std(proportion_predicted[k]) for k in proportion_predicted.keys()], fmt='', color='black')
        thresholds_jitter = np.array([[x] * len(accs[0]) for x in thresholds]).reshape(-1)
        thresholds_jitter = thresholds_jitter + np.random.normal(0, 0.005, thresholds_jitter.shape)
        plt.scatter(
            thresholds_jitter,
            np.stack([np.array(accs[k]) for k in accs.keys()]).reshape(-1),
            color='blue', s=1
        )
        plt.scatter(
            thresholds_jitter,
            np.stack([np.array(mccs[k]) for k in mccs.keys()]).reshape(-1),
            color='red', s=1
        )
        plt.scatter(
            thresholds_jitter,
            np.stack([np.array(proportion_predicted[k]) for k in proportion_predicted.keys()]).reshape(-1),
            color='black', s=1
        )

        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        # insert a dashed line at 0.95
        # plt.xlim(0, 9)
        fig.savefig(f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_thresholds_dots.png')
        run[f'{group}/thresholds'].upload(
            f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_thresholds_dots.png'
        )

        plt.close(fig)
        # save a table of all mccs, accs and FOT for each threshold
        with open(f'{self.log_path}/saved_models/table_{self.args.model_name}_{group}_thresholds.csv', 'w') as f:
            f.write('threshold,acc_mean,acc_std,mcc_mean,mcc_std,FOT,FOT_std\n')
            for thres in thresholds:
                f.write(f'{thres},{np.mean(accs[thres])},{np.std(accs[thres])},'
                        f'{np.mean(mccs[thres])},{np.std(mccs[thres])},'
                        f'{np.mean(proportion_predicted[thres])},{np.std(proportion_predicted[thres])}\n')
        run[f'{group}/thresholds_table'].upload(
            f'{self.log_path}/saved_models/table_{self.args.model_name}_{group}_thresholds.csv'
        )

    def keep_models(self, scaler_name):
        """
        Remove the tmp from the name if the models are to be kept because the best yet
        """
        mode_tag = getattr(self.args, 'mode', 'nomode')
        saved_dir = f'{self.log_path}/saved_models/'
        os.makedirs(saved_dir, exist_ok=True)

        # Determine existing best score (valid_mcc_mean) if present
        existing_score = None
        best_params_path = f'{saved_dir}/best_params_{self.name}_{self.args.model_name}.json'
        if os.path.exists(best_params_path):
            try:
                with open(best_params_path, 'r') as rf:
                    data = json.load(rf)
                    existing_score = data.get('valid_mcc_mean')
            except Exception as e:
                print(f'Could not read existing best params: {e}')

        # New score from current training stored in self.best_scores
        # Compute new_score: best_scores['mcc']['valid'] stored as list after retrieve_best_scores
        bs_valid = self.best_scores['mcc']['valid']
        if isinstance(bs_valid, list) and len(bs_valid) > 0:
            new_score = float(np.mean(bs_valid))
        elif isinstance(bs_valid, (float, int)) and bs_valid is not None:
            new_score = float(bs_valid)
        else:
            new_score = None

        # If existing score is better or equal, discard new tmp artifacts
        if existing_score is not None and new_score is not None and existing_score >= new_score:
            # Remove tmp artifacts produced this round
            for f in list(os.listdir(saved_dir)):
                if f.endswith('_tmp.pkl') and f.startswith(f'{self.args.model_name}_{mode_tag}_{scaler_name}_'):
                    try:
                        os.remove(f'{saved_dir}/{f}')
                    except Exception:
                        pass
            for aux in [f'columns_after_threshold_{scaler_name}_tmp.pkl', f'scaler_{scaler_name}_tmp.pkl']:
                aux_path = f'{saved_dir}/{aux}'
                if os.path.exists(aux_path):
                    try:
                        os.remove(aux_path)
                    except Exception:
                        pass
            print(f'Existing model retained (valid_mcc_mean {existing_score} >= new {new_score}).')
            return False

        # Otherwise promote tmp artifacts to permanent
        for f in os.listdir(saved_dir):
            if f.startswith(f'{self.args.model_name}_{mode_tag}_{scaler_name}_') and f.endswith('_tmp.pkl'):
                final_name = f[:-8] + '.pkl'
                os.rename(f'{saved_dir}/{f}', f'{saved_dir}/{final_name}')
            if f.startswith(f'{self.args.model_name}_{mode_tag}_{scaler_name}_') and f.endswith('_meta_tmp.json'):
                final_name = f[:-13] + '_meta.json'
                os.rename(f'{saved_dir}/{f}', f'{saved_dir}/{final_name}')

        # Promote auxiliary files if present
        aux_map = {
            f'columns_after_threshold_{scaler_name}_tmp.pkl': f'columns_after_threshold_{scaler_name}.pkl',
            f'scaler_{scaler_name}_tmp.pkl': f'scaler_{scaler_name}.pkl'
        }
        for tmp_name, final_name in aux_map.items():
            tmp_path = f'{saved_dir}/{tmp_name}'
            if os.path.exists(tmp_path):
                try:
                    os.rename(tmp_path, f'{saved_dir}/{final_name}')
                except FileExistsError:
                    # Overwrite only if better score (we already established it's better)
                    try:
                        os.remove(f'{saved_dir}/{final_name}')
                        os.rename(tmp_path, f'{saved_dir}/{final_name}')
                    except Exception as e:
                        print(f'Could not replace {final_name}: {e}')
        print(f'New best model saved (valid_mcc_mean {new_score} > {existing_score}).')
        # DVC track promoted artifacts
        try:
            promoted = []
            for f in os.listdir(saved_dir):
                if f.startswith(f'{self.args.model_name}_{mode_tag}_{scaler_name}_') and f.endswith('.pkl'):
                    promoted.append(os.path.join(saved_dir, f))
                if f.startswith(f'{self.args.model_name}_{mode_tag}_{scaler_name}_') and f.endswith('_meta.json'):
                    promoted.append(os.path.join(saved_dir, f))
            any_added = False
            for p in promoted:
                if dvc_add(p):
                    any_added = True
            if any_added:
                # stage .dvc metafiles
                os.system('git add ' + ' '.join(p + '.dvc' for p in promoted if os.path.exists(p + '.dvc')) + ' >/dev/null 2>&1')
                dvc_push()
            # Neptune artifact logging (if run/model available)
            if getattr(self, 'log_neptune', False):
                try:
                    # capture sizes
                    for p in promoted:
                        if os.path.isfile(p):
                            sz = os.path.getsize(p)
                            if hasattr(self, 'model_version_ref'):
                                self.model_version_ref[f'artifacts/{os.path.basename(p)}/bytes'] = int(sz)
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: DVC tracking of best model failed: {e}")

        return True

    def remove_models(self, scaler_name: str):
        try:
            with suppress(FileNotFoundError):
                os.remove(f'{self.log_path}/saved_models/scaler_{scaler_name}_tmp.pkl')
        except Exception as e:
            print(f"Error removing scaler tmp file: {e}")
        try:
            with suppress(FileNotFoundError):
                os.remove(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl')
        except Exception as e:
            print(f"Error removing columns_after_threshold tmp file: {e}")
        try:
            with suppress(FileNotFoundError):
                os.remove(f'{self.log_path}/saved_models/{self.args.model_name}_state_tmp.pth')
        except Exception as e:
            print(f"Error removing model state tmp file: {e}")
        try:
            with suppress(FileNotFoundError):
                os.remove(f'{self.log_path}/saved_models/{self.args.model_name}_tmp.pth')
        except Exception as e:
            print(f"Error removing model tmp file: {e}")

    def dump_models(self, models, scaler_name, lists):
        # Save unique labels
        with open(f'{self.log_path}/saved_models/unique_labels.json', "w") as read_file:
            json.dump(self.unique_labels.tolist(), read_file)

        for i, m in enumerate(models):
            # save model
            with open(
                f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}.pkl',
                 'wb') as f:
                pickle.dump(m, f)
            # save indices
            with open(
                f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_train_indices.pkl',
                 'wb') as f:
                pickle.dump(lists['inds']['train'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_valid_indices.pkl',
                      'wb') as f:
                pickle.dump(lists['inds']['valid'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_test_indices.pkl',
                      'wb') as f:
                pickle.dump(lists['inds']['test'][i], f)

    def dump_model(self, i, m, scaler_name, lists):
        # Save unique labels
        with open(f'{self.log_path}/saved_models/unique_labels.json', "w") as read_file:
            json.dump(self.unique_labels.tolist(), read_file)

        # save model
        mode_tag = getattr(self.args, 'mode', 'nomode')
        with open(f'{self.log_path}/saved_models/{self.args.model_name}_{mode_tag}_{scaler_name}_{i}_tmp.pkl', 'wb') as f:
            pickle.dump(m, f)
        # metadata snapshot (temporary). Best model promotion will rename _meta_tmp.json
        try:
            meta = {
                'model_name': self.args.model_name,
                'mode': mode_tag,
                'scaler': scaler_name,
                'repeat_index': i,
                'unique_labels': self.unique_labels.tolist(),
                'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
                'args': {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                         for k, v in vars(self.args).items() if not k.startswith('_')},
                'hparams': getattr(self, 'best_params_dict', {}),
                'metrics_snapshot': {
                    'train_acc_mean': float(np.mean(self.best_scores['acc']['train'])) if self.best_scores['acc']['train'] else None,
                    'valid_acc_mean': float(np.mean(self.best_scores['acc']['valid'])) if self.best_scores['acc']['valid'] else None,
                    'test_acc_mean': float(np.mean(self.best_scores['acc']['test'])) if self.best_scores['acc']['test'] else None,
                    'train_mcc_mean': float(np.mean(self.best_scores['mcc']['train'])) if self.best_scores['mcc']['train'] else None,
                    'valid_mcc_mean': float(np.mean(self.best_scores['mcc']['valid'])) if self.best_scores['mcc']['valid'] else None,
                    'test_mcc_mean': float(np.mean(self.best_scores['mcc']['test'])) if self.best_scores['mcc']['test'] else None,
                }
            }
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{mode_tag}_{scaler_name}_{i}_meta_tmp.json', 'w') as jf:
                json.dump(meta, jf, indent=2)
        except Exception as e:
            print(f'Warning: failed to write metadata for model repeat {i}: {e}')

        if lists is not None:
            # save indices
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{mode_tag}_{scaler_name}_{i}_train_indices_tmp.pkl',
                      'wb') as f:
                pickle.dump(lists['inds']['train'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{mode_tag}_{scaler_name}_{i}_valid_indices_tmp.pkl',
                      'wb') as f:
                pickle.dump(lists['inds']['valid'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{mode_tag}_{scaler_name}_{i}_test_indices_tmp.pkl',
                      'wb') as f:
                pickle.dump(lists['inds']['test'][i], f)

    def retrieve_best_scores(self, lists):
        self.best_scores['acc']['train'] = lists['acc']['train']
        self.best_scores['acc']['valid'] = lists['acc']['valid']
        self.best_scores['acc']['test'] = lists['acc']['test']
        self.best_scores['mcc']['train'] = lists['mcc']['train']
        self.best_scores['mcc']['valid'] = lists['mcc']['valid']
        self.best_scores['mcc']['test'] = lists['mcc']['test']

    def save_confusion_matrices(self, lists, run):
        relevant_samples = []
        if len(lists['preds']['posurines']) > 0:
            posurines_df, lists = self.make_predictions(lists, run)
            relevant_samples = [
                i for i, l in enumerate(posurines_df.loc[:, 'labels'].to_numpy()) if l in self.unique_labels
            ]
            if len(relevant_samples) > 0:
                posurines_df = posurines_df.iloc[relevant_samples]
                posurines_classes = [
                    int(
                        np.argwhere(label == self.unique_labels).flatten()
                    ) for label in posurines_df.loc[:, 'labels'].to_numpy()
                ]
                posurines_preds = [
                    int(
                        np.argwhere(label == self.unique_labels).flatten()
                    ) for label in posurines_df.loc[:, 'preds'].to_numpy()
                ]

                lists['acc']['posurines'] = [ACC(posurines_preds, posurines_classes)]
                lists['mcc']['posurines'] = [MCC(posurines_preds, posurines_classes)]
                lists['classes']['posurines'] = [posurines_classes for _ in range(len(lists['preds']['posurines']))]
                lists['preds']['posurines'] = [posurines_preds for _ in range(len(lists['preds']['posurines']))]
                lists['names']['posurines'] =\
                    [posurines_df.loc[:, 'names'].to_numpy() for _ in range(len(lists['preds']['posurines']))]
                lists['batches']['posurines'] =\
                    [posurines_df.loc[:, 'batches'].to_numpy() for _ in range(len(lists['preds']['posurines']))]
                lists['labels']['posurines'] =\
                    [posurines_df.loc[:, 'labels'].to_numpy() for _ in range(len(lists['preds']['posurines']))]
                lists['proba']['posurines'] =\
                    [posurines_df.loc[:, 'proba'].to_numpy() for _ in range(len(lists['preds']['posurines']))]

                fig = get_confusion_matrix(posurines_classes, posurines_preds, self.unique_labels)
                save_confusion_matrix(fig,
                                      f"{self.log_path}/confusion_matrices/"
                                      f"{self.name}_{self.args.model_name}_posurines",
                                      acc=lists['acc']['posurines'], mcc=lists['mcc']['posurines'], group='posurines')
                run['confusion_matrix/posurines'].upload(fig)
        else:
            posurines_df = pd.DataFrame([])
        groups = ['train', 'valid', 'test']
        for group in groups:
            fig = get_confusion_matrix(np.concatenate(lists['classes'][group]),
                                       np.concatenate(lists['preds'][group]),
                                       self.unique_labels)
            save_confusion_matrix(fig,
                                  f"{self.log_path}/confusion_matrices/"
                                  f"{self.name}_{self.args.model_name}_{group}",
                                  acc=lists['acc'][group], mcc=lists['mcc'][group], group=group)
            run[f'confusion_matrix/{group}'].upload(fig)

        return lists, posurines_df
