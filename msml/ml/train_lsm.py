import sys
import copy
# import json
import os
import neptune
import pickle
import numpy as np
import pandas as pd
from bernn.dl.train.pytorch.utils.utils import LogConfusionMatrix
from .utils import remove_zero_cols, scale_data, get_empty_lists, columns_stats_over0
from .torch_utils import augment_data
from .loggings import log_ord, log_fct, log_neptune
# from sklearn_train_nocv import get_confusion_matrix, plot_roc
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
from .log_shap import log_shap
# import matplotlib.pyplot as plt
# import pipeline from sklearn
from bernn.dl.train.train_ae import TrainAE
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
# import xgboost
from bernn.dl.train.pytorch.utils.dataset import get_loaders_bacteria
# from bernn.ml.train.params_gp import linsvc_space
from bernn.dl.train.train_ae_classifier_holdout import log_num_neurons
# from bernn.utils.utils import scale_data
from bernn.dl.train.pytorch.utils.utils import get_optimizer, get_empty_dicts, get_empty_traces, \
    log_traces, get_best_values, add_to_logger, add_to_neptune, add_to_mlflow
# from torch import nn
import torch
# from sklearn.calibration import calibration_curve
from .train import Train
from msml.ml.models.train_lsm import TrainLSM

# from tensorboardX import SummaryWriter

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'


def get_size_in_mb(obj):
    size_in_bytes = sys.getsizeof(obj)
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb


def score(predictions, labels):
    return ACC(labels, predictions)


class Train_bernn(TrainAE, Train):
    def __init__(self, name, model, data, uniques,
                 log_path, args, log_metrics, logger, log_neptune, mlops='None',
                 binary_path=None):
        # Initialize TrainAE first
        TrainAE.__init__(self, args, path=None, fix_thres=-1, load_tb=False, log_metrics=log_metrics,
                         keep_models=False, log_inputs=False, log_plots=False, log_tb=False,
                         log_neptune=log_neptune, log_mlflow=False, groupkfold=args.groupkfold,
                         pools=False)
        # Initialize Train second
        Train.__init__(self, name, model, data, uniques, log_path, args, logger, log_neptune, mlops,
                       log_metrics, binary_path)

        os.makedirs(log_path, exist_ok=True)
        # self.n_repeats = 5
        self.n_agg = 1
        self.early_stop = 100
        self.train_after_warmup = True
        self.predict_tests = False
        self.best_mcc = -1
        self.best_mccs = []  # Initialize best_mccs list
        self.best_closses = []  # Initialize best_closses list

        self.complete_log_path = os.path.join(log_path, 'logs')
        os.makedirs(self.complete_log_path, exist_ok=True)
        os.makedirs('logs/ae', exist_ok=True)
        self.best_loss = 1e10
        self.best_iteration = 0

    def load_autoencoder(self):
        if not self.args.kan:
            from bernn import AutoEncoder3 as AutoEncoder
            from bernn import SHAPAutoEncoder3 as SHAPAutoEncoder
        elif self.args.kan == 1:
            from bernn import KANAutoencoder3 as AutoEncoder
            from bernn import SHAPKANAutoencoder3 as SHAPAutoEncoder
        self.ae = AutoEncoder
        self.shap_ae = SHAPAutoEncoder

    def get_data(self, all_data, h):
        data = {}
        info_keys = ['inputs', 'names', 'labels', 'cats', 'batches',
                     'batches_labels', 'manips', 'orders', 'sets', 'urines', 'concs']
        for info in info_keys:
            data[info] = {}
            for group in ['train', 'valid', 'test', 'all', 'urinespositives']:
                data[info][group] = np.array([])

        if self.args.train_on == 'all':
            if self.args.groupkfold:
                skf = StratifiedGroupKFold(
                    n_splits=len(np.unique(all_data['batches']['all'])),
                    shuffle=True, random_state=42
                )
                train_nums = np.arange(0, len(all_data['labels']['all']))
                splitter = skf.split(
                    train_nums, 
                    all_data['labels']['all'],
                    all_data['batches']['all']
                )
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                train_nums = np.arange(0, len(all_data['labels']['all']))
                splitter = skf.split(train_nums, all_data['labels']['all'])

                # Split urinespositives
                skf_upos = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                train_nums_upos = np.arange(0, len(all_data['labels']['urinespositives']))
                splitter_upos = skf_upos.split(
                    train_nums_upos,
                    all_data['labels']['urinespositives']
                )

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

            # Create data dictionary with indices
            # all_data['cats'] = {
            #     'train': np.array([
            # np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['all'][train_inds]]),
            #     'valid': np.array([
            # np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['all'][valid_inds]]),
            #     'test': np.array([
            # np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['all'][test_inds]])
            # }
            # all_data['meta'] = {
            #     'train': all_data['inputs']['all'].iloc[train_inds, :2],
            #     'valid': all_data['inputs']['all'].iloc[valid_inds, :2],
            #     'test': all_data['inputs']['all'].iloc[test_inds, :2]
            # }
            data = {
                'inputs': {
                    'train': all_data['inputs']['all'].iloc[train_inds],
                    'valid': all_data['inputs']['all'].iloc[valid_inds],
                    'test': all_data['inputs']['all'].iloc[test_inds],
                    'all': all_data['inputs']['all'],
                    'urinespositives': all_data['inputs']['urinespositives']
                },
                'labels': {
                    'train': all_data['labels']['all'][train_inds],
                    'valid': all_data['labels']['all'][valid_inds],
                    'test': all_data['labels']['all'][test_inds],
                    'all': all_data['labels']['all'],
                    'urinespositives': all_data['labels']['urinespositives']
                },
                'batches': {
                    'train': all_data['batches']['all'][train_inds],
                    'valid': all_data['batches']['all'][valid_inds],
                    'test': all_data['batches']['all'][test_inds],
                    'all': all_data['batches']['all'],
                    'urinespositives': all_data['batches']['urinespositives']
                },
                'batches_labels': {
                    'train': all_data['batches_labels']['all'][train_inds],
                    'valid': all_data['batches_labels']['all'][valid_inds],
                    'test': all_data['batches_labels']['all'][test_inds],
                    'all': all_data['batches_labels']['all'],
                    'urinespositives': all_data['batches_labels']['urinespositives']
                },
                'names': {
                    'train': all_data['names']['all'][train_inds],
                    'valid': all_data['names']['all'][valid_inds],
                    'test': all_data['names']['all'][test_inds],
                    'all': all_data['names']['all'],
                    'urinespositives': all_data['names']['urinespositives']
                },
                'sets': {
                    'train': np.array(['train'] * len(train_inds)),
                    'valid': np.array(['valid'] * len(valid_inds)),
                    'test': np.array(['test'] * len(test_inds)),
                    'all': np.array(['all'] * len(all_data['inputs']['all'])),
                    'urinespositives': np.array(['urinespositives'] * len(all_data['inputs']['urinespositives']))
                },
                'cats': {
                    'train': all_data['cats']['all'][train_inds],
                    'valid': all_data['cats']['all'][valid_inds],
                    'test': all_data['cats']['all'][test_inds],
                    'all': all_data['cats']['all'],
                    'urinespositives': all_data['cats']['urinespositives']
                },
                'meta': {
                    'train': all_data['meta']['all'].iloc[train_inds],
                    'valid': all_data['meta']['all'].iloc[valid_inds],
                    'test': all_data['meta']['all'].iloc[test_inds],
                    'all': all_data['meta']['all'],
                    'urinespositives': all_data['meta']['urinespositives']
                },
                'urines': {
                    'train': all_data['urines']['all'][train_inds],
                    'valid': all_data['urines']['all'][valid_inds],
                    'test': all_data['urines']['all'][test_inds],
                    'all': all_data['urines']['all'],
                    'urinespositives': all_data['urines']['urinespositives']
                },
                'concs': {
                    'train': all_data['concs']['all'][train_inds],
                    'valid': all_data['concs']['all'][valid_inds],
                    'test': all_data['concs']['all'][test_inds],
                    'all': all_data['concs']['all'],
                    'urinespositives': all_data['concs']['urinespositives']
                },
                'manips': {
                    'train': all_data['manips']['all'][train_inds],
                    'valid': all_data['manips']['all'][valid_inds],
                    'test': all_data['manips']['all'][test_inds],
                    'all': all_data['manips']['all'],
                    'urinespositives': all_data['manips']['urinespositives']
                },
                'inds': {
                    'train': train_inds,
                    'valid': valid_inds,
                    'test': test_inds,
                }
            }

            if not self.args.groupkfold:
                train_inds_upos =\
                      [x for x in train_nums_upos if x not in np.concatenate((valid_inds_upos, test_inds_upos))]

                # Add urinespositives data
                train_data = pd.concat((
                    data['inputs']['train'],
                    all_data['inputs']['urinespositives'].iloc[train_inds_upos]
                ), axis=0)
                valid_data = pd.concat((
                    data['inputs']['valid'],
                    all_data['inputs']['urinespositives'].iloc[valid_inds_upos]
                ), axis=0)

                train_meta = pd.concat((
                    data['meta']['train'],
                    all_data['meta']['urinespositives'].iloc[train_inds_upos]
                ), axis=0)
                valid_meta = pd.concat((
                    data['meta']['valid'],
                    all_data['meta']['urinespositives'].iloc[valid_inds_upos]
                ), axis=0)

                train_labels = np.concatenate((
                    data['labels']['train'],
                    all_data['labels']['urinespositives'][train_inds_upos]
                ))
                valid_labels = np.concatenate((
                    data['labels']['valid'],
                    all_data['labels']['urinespositives'][valid_inds_upos]
                ))

                train_batches = np.concatenate((
                    data['batches']['train'],
                    all_data['batches']['urinespositives'][train_inds_upos]
                ))
                valid_batches = np.concatenate((
                    data['batches']['valid'],
                    all_data['batches']['urinespositives'][valid_inds_upos]
                ))

                train_batches_labels = np.concatenate((
                    data['batches_labels']['train'],
                    all_data['batches_labels']['urinespositives'][train_inds_upos]
                ))
                valid_batches_labels = np.concatenate((
                    data['batches_labels']['valid'],
                    all_data['batches_labels']['urinespositives'][valid_inds_upos]
                ))

                train_names = np.concatenate((
                    data['names']['train'],
                    all_data['names']['urinespositives'][train_inds_upos]
                ))
                valid_names = np.concatenate((
                    data['names']['valid'],
                    all_data['names']['urinespositives'][valid_inds_upos]
                ))

                train_sets = np.concatenate((
                    data['sets']['train'],
                    ['train' for _ in train_inds_upos]
                ))
                valid_sets = np.concatenate((
                    data['sets']['valid'],
                    ['valid' for _ in train_inds_upos]
                ))

                train_manips = np.concatenate((
                    data['manips']['train'],
                    all_data['manips']['urinespositives'][train_inds_upos]
                ))
                valid_manips = np.concatenate((
                    data['manips']['valid'],
                    all_data['manips']['urinespositives'][valid_inds_upos]
                ))

                train_urines = np.concatenate((
                    data['urines']['train'],
                    all_data['urines']['urinespositives'][train_inds_upos]
                ))
                valid_urines = np.concatenate((
                    data['urines']['valid'],
                    all_data['urines']['urinespositives'][valid_inds_upos]
                ))

                train_concs = np.concatenate((
                    data['concs']['train'],
                    all_data['concs']['urinespositives'][train_inds_upos]
                ))
                valid_concs = np.concatenate((
                    data['concs']['valid'],
                    all_data['concs']['urinespositives'][valid_inds_upos]
                ))

                train_cats = np.concatenate((
                    data['cats']['train'],
                    all_data['cats']['urinespositives'][train_inds_upos]
                ))
                valid_cats = np.concatenate((
                    data['cats']['valid'],
                    all_data['cats']['urinespositives'][valid_inds_upos]
                ))

                # Update data dictionary
                data['inputs']['train'] = train_data
                data['inputs']['valid'] = valid_data
                data['labels']['train'] = train_labels
                data['labels']['valid'] = valid_labels
                data['batches']['train'] = train_batches
                data['batches']['valid'] = valid_batches
                data['batches_labels']['train'] = train_batches_labels
                data['batches_labels']['valid'] = valid_batches_labels
                data['names']['train'] = train_names
                data['names']['valid'] = valid_names

                data['cats']['train'] = train_cats
                data['cats']['valid'] = valid_cats
                data['meta']['train'] = train_meta
                data['meta']['valid'] = valid_meta
                data['batches']['train'] = train_batches
                data['batches']['valid'] = valid_batches
                data['urines']['train'] = train_urines
                data['urines']['valid'] = valid_urines
                data['concs']['train'] = train_concs
                data['concs']['valid'] = valid_concs
                data['sets']['train'] = train_sets
                data['sets']['valid'] = valid_sets
                data['manips']['train'] = train_manips
                data['manips']['valid'] = valid_manips

                # Add urinespositives indices
                data['inds']['train_upos'] = train_inds_upos
                data['inds']['valid_upos'] = valid_inds_upos
                data['inds']['test_upos'] = test_inds_upos

                # Drop urinespositives from train and valid
                b_inds = np.concatenate((train_inds_upos, valid_inds_upos)).astype(int)
                all_data['labels']['urinespositives'] = np.delete(
                    all_data['labels']['urinespositives'],
                    b_inds
                )
                all_data['batches']['urinespositives'] = np.delete(
                    all_data['batches']['urinespositives'],
                    b_inds
                )
                all_data['batches_labels']['urinespositives'] = np.delete(
                    all_data['batches_labels']['urinespositives'],
                    b_inds
                )
                all_data['names']['urinespositives'] = np.delete(
                    all_data['names']['urinespositives'],
                    b_inds
                )
                all_data['concs']['urinespositives'] = np.delete(
                    all_data['concs']['urinespositives'],
                    b_inds
                )
                all_data['cats']['urinespositives'] = np.delete(
                    all_data['cats']['urinespositives'],
                    b_inds
                )
                all_data['urines']['urinespositives'] = np.delete(
                    all_data['urines']['urinespositives'],
                    b_inds
                )
                # all_data['sets']['urinespositives'] = np.array([
                # 'urinespositives' for _ in all_data['names']['urinespositives']
                # ])
                all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'].loc[
                    all_data['names']['urinespositives'], :
                ]
                all_data['meta']['urinespositives'] = all_data['meta']['urinespositives'].loc[
                    all_data['names']['urinespositives'], :
                ]
                all_data['manips']['urinespositives'] = all_data['manips']['urinespositives'][b_inds]
                all_data['meta']['urinespositives'] = all_data['inputs']['urinespositives'].iloc[:, :2]

            # Remove labels that are not in train
            unique_train_labels = np.unique(data['labels']['train'])
            valid_to_keep = np.array([i for i, l in enumerate(data['labels']['valid']) if l in unique_train_labels])
            test_to_keep = np.array([i for i, l in enumerate(data['labels']['test']) if l in unique_train_labels])

            data['inputs']['valid'] = data['inputs']['valid'].iloc[valid_to_keep]
            data['labels']['valid'] = data['labels']['valid'][valid_to_keep]
            data['batches']['valid'] = data['batches']['valid'][valid_to_keep]
            data['batches_labels']['valid'] = data['batches_labels']['valid'][valid_to_keep]
            data['names']['valid'] = data['names']['valid'][valid_to_keep]
            data['sets']['valid'] = data['sets']['valid'][valid_to_keep]
            data['cats']['valid'] = data['cats']['valid'][valid_to_keep]
            data['meta']['valid'] = data['meta']['valid'].iloc[valid_to_keep]
            data['urines']['valid'] = data['urines']['valid'][valid_to_keep]
            data['concs']['valid'] = data['concs']['valid'][valid_to_keep]
            data['manips']['valid'] = data['manips']['valid'][valid_to_keep]

            data['inputs']['test'] = data['inputs']['test'].iloc[test_to_keep]
            data['labels']['test'] = data['labels']['test'][test_to_keep]
            data['batches']['test'] = data['batches']['test'][test_to_keep]
            data['batches_labels']['test'] = data['batches_labels']['test'][test_to_keep]
            data['names']['test'] = data['names']['test'][test_to_keep]
            data['sets']['test'] = data['sets']['test'][test_to_keep]
            data['cats']['test'] = data['cats']['test'][test_to_keep]
            data['meta']['test'] = data['meta']['test'].iloc[test_to_keep]
            data['urines']['test'] = data['urines']['test'][test_to_keep]
            data['concs']['test'] = data['concs']['test'][test_to_keep]
            data['manips']['test'] = data['manips']['test'][test_to_keep]

        elif self.args.train_on == 'all_lows':
            # Keep all low concentration samples for training
            train_inds = np.argwhere(all_data['concs']['all'] == 'l').flatten()
            valid_inds = np.argwhere(all_data['concs']['all'] == 'h').flatten()
            blanc_inds = np.argwhere(all_data['concs']['all'] == 'na').flatten()

            data = {
                'inputs': {
                    'train': all_data['inputs']['all'].iloc[train_inds],
                    'valid': all_data['inputs']['all'].iloc[valid_inds],
                    'blanc': all_data['inputs']['all'].iloc[blanc_inds]
                },
                'labels': {
                    'train': all_data['labels']['all'][train_inds],
                    'valid': all_data['labels']['all'][valid_inds],
                    'blanc': all_data['labels']['all'][blanc_inds]
                },
                'batches': {
                    'train': all_data['batches']['all'][train_inds],
                    'valid': all_data['batches']['all'][valid_inds],
                    'blanc': all_data['batches']['all'][blanc_inds]
                },
                'names': {
                    'train': all_data['names']['all'][train_inds],
                    'valid': all_data['names']['all'][valid_inds],
                    'blanc': all_data['names']['all'][blanc_inds]
                },
                'inds': {
                    'train': train_inds,
                    'valid': valid_inds,
                    'blanc': blanc_inds
                }
            }

            # Split blanc samples
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            blanc_nums = np.arange(0, len(data['labels']['blanc']))
            splitter = skf.split(blanc_nums, data['labels']['blanc'], data['batches']['blanc'])
            blanc_train_inds, blanc_valid_inds = splitter.__next__()

            # Update data with blanc splits
            data['inputs']['train'] = pd.concat((
                data['inputs']['train'], data['inputs']['blanc'].iloc[blanc_train_inds]
            ))
            data['inputs']['valid'] = pd.concat((
                data['inputs']['valid'], data['inputs']['blanc'].iloc[blanc_valid_inds]
            ))

            data['labels']['train'] = np.concatenate((
                data['labels']['train'], data['labels']['blanc'][blanc_train_inds]
            ))
            data['labels']['valid'] = np.concatenate((
                data['labels']['valid'], data['labels']['blanc'][blanc_valid_inds]
            ))

            data['batches']['train'] = np.concatenate((
                data['batches']['train'], data['batches']['blanc'][blanc_train_inds]
            ))
            data['batches']['valid'] = np.concatenate((
                data['batches']['valid'], data['batches']['blanc'][blanc_valid_inds]
            ))

            data['names']['train'] = np.concatenate((data['names']['train'], data['names']['blanc'][blanc_train_inds]))
            data['names']['valid'] = np.concatenate((data['names']['valid'], data['names']['blanc'][blanc_valid_inds]))

            # Add blanc indices
            data['inds']['blanc_train'] = blanc_train_inds
            data['inds']['blanc_valid'] = blanc_valid_inds

            # Split validation into validation and test
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            valid_nums = np.arange(0, len(data['labels']['valid']))
            splitter = skf.split(valid_nums, data['labels']['valid'], data['batches']['valid'])
            test_inds, valid_inds = splitter.__next__()

            # Update data with test split
            data['inputs']['test'] = data['inputs']['valid'].iloc[test_inds]
            data['inputs']['valid'] = data['inputs']['valid'].iloc[valid_inds]

            data['labels']['test'] = data['labels']['valid'][test_inds]
            data['labels']['valid'] = data['labels']['valid'][valid_inds]

            data['batches']['test'] = data['batches']['valid'][test_inds]
            data['batches']['valid'] = data['batches']['valid'][valid_inds]

            data['names']['test'] = data['names']['valid'][test_inds]
            data['names']['valid'] = data['names']['valid'][valid_inds]

            # Add test indices
            data['inds']['test'] = test_inds
            data['inds']['valid'] = valid_inds

        elif self.args.train_on == 'all_highs':
            # Keep all high concentration samples for training
            train_inds = np.argwhere(all_data['concs']['all'] == 'h').flatten()
            valid_inds = np.argwhere(all_data['concs']['all'] == 'l').flatten()
            blanc_inds = np.argwhere(all_data['concs']['all'] == 'na').flatten()

            data = {
                'inputs': {
                    'train': all_data['inputs']['all'].iloc[train_inds],
                    'valid': all_data['inputs']['all'].iloc[valid_inds],
                    'blanc': all_data['inputs']['all'].iloc[blanc_inds]
                },
                'labels': {
                    'train': all_data['labels']['all'][train_inds],
                    'valid': all_data['labels']['all'][valid_inds],
                    'blanc': all_data['labels']['all'][blanc_inds]
                },
                'batches': {
                    'train': all_data['batches']['all'][train_inds],
                    'valid': all_data['batches']['all'][valid_inds],
                    'blanc': all_data['batches']['all'][blanc_inds]
                },
                'names': {
                    'train': all_data['names']['all'][train_inds],
                    'valid': all_data['names']['all'][valid_inds],
                    'blanc': all_data['names']['all'][blanc_inds]
                },
                'inds': {
                    'train': train_inds,
                    'valid': valid_inds,
                    'blanc': blanc_inds
                }
            }

            # Split blanc samples
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            blanc_nums = np.arange(0, len(data['labels']['blanc']))
            splitter = skf.split(blanc_nums, data['batches']['blanc'], data['batches']['blanc'])
            blanc_train_inds, blanc_valid_inds = splitter.__next__()

            # Update data with blanc splits
            data['inputs']['train'] = pd.concat((
                data['inputs']['train'], data['inputs']['blanc'].iloc[blanc_train_inds]
            ))
            data['inputs']['valid'] = pd.concat((
                data['inputs']['valid'], data['inputs']['blanc'].iloc[blanc_valid_inds]
            ))

            data['labels']['train'] = np.concatenate((
                data['labels']['train'], data['labels']['blanc'][blanc_train_inds]
            ))
            data['labels']['valid'] = np.concatenate((
                data['labels']['valid'], data['labels']['blanc'][blanc_valid_inds]
            ))

            data['batches']['train'] = np.concatenate((
                data['batches']['train'], data['batches']['blanc'][blanc_train_inds]
            ))
            data['batches']['valid'] = np.concatenate((
                data['batches']['valid'], data['batches']['blanc'][blanc_valid_inds]
            ))

            data['names']['train'] = np.concatenate((data['names']['train'], data['names']['blanc'][blanc_train_inds]))
            data['names']['valid'] = np.concatenate((data['names']['valid'], data['names']['blanc'][blanc_valid_inds]))

            # Add blanc indices
            data['inds']['blanc_train'] = blanc_train_inds
            data['inds']['blanc_valid'] = blanc_valid_inds

            # Split validation into validation and test
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
            valid_nums = np.arange(0, len(data['labels']['valid']))
            splitter = skf.split(valid_nums, data['batches']['valid'], data['batches']['valid'])
            test_inds, valid_inds = splitter.__next__()

            # Update data with test split
            data['inputs']['test'] = data['inputs']['valid'].iloc[test_inds]
            data['inputs']['valid'] = data['inputs']['valid'].iloc[valid_inds]

            data['labels']['test'] = data['labels']['valid'][test_inds]
            data['labels']['valid'] = data['labels']['valid'][valid_inds]

            data['batches']['test'] = data['batches']['valid'][test_inds]
            data['batches']['valid'] = data['batches']['valid'][valid_inds]

            data['names']['test'] = data['names']['valid'][test_inds]
            data['names']['valid'] = data['names']['valid'][valid_inds]

            # Add test indices
            data['inds']['test'] = test_inds
            data['inds']['valid'] = valid_inds

        unique_labels = np.unique(data['labels']['all'])
        unique_batches = np.unique(np.concatenate((data['batches']['all'], data['batches']['urinespositives'])))
        return data, unique_labels, unique_batches

    def train_bernn(self, all_data, h, params, run):
        # TODO these params should not be set here
        metrics = {}
        self.best_mcc = -1
        self.warmup_counter = 0
        self.warmup_b_counter = 0
        self.warmup_disc_b = False
        # params['warmup'] = 30       print(self.foldername)
        warmup = True
        self.columns = all_data['inputs']['all'].columns
        h += 1
        self.make_samples_weights()

        # Transform the data with the chosen scaler
        data = copy.deepcopy(all_data)
        data, self.scaler = scale_data(params['scaler'], data, self.args.device)

        # feature_selection = get_feature_selection_method('mutual_info_classif')
        # mi = feature_selection(data['inputs']['train'], data['cats']['train'])
        for g in list(data['inputs'].keys()):
            data['inputs'][g] = data['inputs'][g].round(4)
        # Gets all the pytorch dataloaders to train the models
        loaders = get_loaders_bacteria(data, self.args.random_recs, self.samples_weights, self.args.dloss,
                                       None, None, bs=self.args.bs)

        ae = self.ae(data['inputs']['all'].shape[1],
                     n_batches=self.n_batches,
                     nb_classes=self.n_cats,
                     mapper=self.args.use_mapping,
                     layers=self.get_ordered_layers(params),
                     n_layers=self.args.n_layers,
                     n_meta=self.args.n_meta,
                     n_emb=self.args.embeddings_meta,
                     dropout=params['dropout'],
                     variational=self.args.variational, conditional=False,
                     zinb=self.args.zinb, add_noise=0, tied_weights=self.args.tied_weights,
                     use_gnn=0,  # TODO to remove
                     device=self.args.device,
                     prune_threshold=self.args.prune_threshold,
                     update_grid=self.args.update_grid
                     ).to(self.args.device)
        ae.mapper.to(self.args.device)
        ae.dec.to(self.args.device)
        n_neurons = ae.prune_model_paperwise(False, False, weight_threshold=self.args.prune_threshold)
        init_n_neurons = ae.count_n_neurons()
        shap_ae = self.shap_ae(data['inputs']['all'].shape[1],
                               n_batches=self.n_batches,
                               nb_classes=self.n_cats,
                               mapper=self.args.use_mapping,
                               layers=self.get_ordered_layers(params),
                               n_layers=self.args.n_layers,
                               n_meta=self.args.n_meta,
                               n_emb=self.args.embeddings_meta,
                               dropout=params['dropout'],
                               variational=self.args.variational, conditional=False,
                               zinb=self.args.zinb, add_noise=0, tied_weights=self.args.tied_weights,
                               use_gnn=0,  # TODO remove this
                               device=self.args.device).to(self.args.device)
        shap_ae.mapper.to(self.args.device)
        shap_ae.dec.to(self.args.device)
        sceloss, celoss, mseloss, triplet_loss = self.get_losses(
            params['scaler'], params['smoothing'], params['margin'], self.args.dloss
        )

        optimizer_ae = get_optimizer(ae, params['lr'], params['wd'], 'adam')

        # Used only if bdisc==1
        optimizer_b = get_optimizer(ae.dann_discriminator, 1e-2, 0, 'adam')

        # self.hparams_names = [x.name for x in linsvc_space]
        if self.log_inputs and not self.logged_inputs:
            data['inputs']['all'].to_csv(
                f'{self.complete_log_path}/{self.args.berm}_inputs.csv')
            if self.args.log_neptune:
                run["inputs.csv"].track_files(f'{self.complete_log_path}/{self.args.berm}_inputs.csv')
            # log_input_ordination(loggers['logger'], data, self.scaler, epoch)
            # if self.pools:
            #     metrics = log_pool_metrics(data['inputs'], data['batches'], data['labels'], loggers, epoch,
            #                                 metrics, 'inputs')
            self.logged_inputs = True

        values, best_values, _, best_traces = get_empty_dicts()

        early_stop_counter = 0
        best_vals = values
        loggers = {'cm_logger': LogConfusionMatrix(self.complete_log_path)}
        if h > 1:  # or warmup_counter == 100:
            ae.load_state_dict(torch.load(f'{self.complete_log_path}/warmup.pth'))
            print("\n\nNO WARMUP\n\n")
        if h == 1:
            for epoch in range(0, self.args.warmup):
                no_error, ae = self.warmup_loop(optimizer_ae, ae, celoss, loaders['all'], triplet_loss, mseloss,
                                                warmup, epoch, optimizer_b, values, loggers, loaders, run,
                                                self.args.use_mapping)
                if not no_error:
                    break
        for epoch in range(0, self.args.n_epochs):
            if early_stop_counter == self.args.early_stop:
                if self.verbose > 0:
                    print('EARLY STOPPING.', epoch)
                break
            lists, traces = get_empty_traces()
            # Copy list test and make it urinespositives
            lists['urinespositives'] = lists['test']
            traces['urinespositives'] = traces['test']
            values['urinespositives'] = values['test']

            if self.args.warmup_after_warmup:
                no_error, ae = self.warmup_loop(optimizer_ae, ae, celoss, loaders['all'], triplet_loss, mseloss,
                                                0, epoch,
                                                optimizer_b, values, loggers, loaders, run, self.args.use_mapping)
                if not no_error:
                    break
            if not self.args.train_after_warmup:
                ae = self.freeze_all_but_clayers(ae)
            closs, _, _ = self.loop('train', optimizer_ae, ae, sceloss,
                                    loaders['train'], lists, traces, nu=params['nu'])

            if torch.isnan(closs):
                print("NAN LOSS")
                break
            ae.eval()
            ae.mapper.eval()

            # Below is the loop for all sets
            with torch.no_grad():
                for group in list(data['inputs'].keys()):
                    if group in ['all', 'all_pool']:
                        continue
                    closs, lists, traces = self.loop(group, optimizer_ae, ae, sceloss,
                                                     loaders[group], lists, traces, nu=0)
                closs, _, _ = self.loop('train', optimizer_ae, ae, sceloss,
                                        loaders['train'], lists, traces, nu=0)

            traces = self.get_mccs(lists, traces)
            values.pop('all_pool', None)
            values.pop('train_pool', None)
            values.pop('valid_pool', None)
            values.pop('test_pool', None)
            values.pop('urinespositives_pool', None)
            traces.pop('all_pool', None)
            traces.pop('train_pool', None)
            traces.pop('valid_pool', None)
            traces.pop('test_pool', None)
            traces.pop('urinespositives_pool', None)
            values = log_traces(traces, values)
            # Remove pool metrics
            if self.args.log_tb:
                try:
                    add_to_logger(values, loggers['logger'], epoch)
                except Exception as e:
                    print(f"Problem with add_to_logger: {e}")
            if self.args.log_neptune:
                add_to_neptune(run, values)
            if self.args.log_mlflow:
                add_to_mlflow(values, epoch)
            if np.mean(values['valid']['mcc'][-self.args.n_agg:]) > self.best_mcc and len(
                    values['valid']['mcc']) >= self.args.n_agg:
                print(f"Best Classification Mcc Epoch {epoch}, "
                      f"Acc: {values['test']['acc'][-1]}"
                      f"VALID Mcc: {values['valid']['mcc'][-1]}"
                      f"TEST Mcc: {values['test']['mcc'][-1]}"
                      f"Classification train loss: {values['train']['closs'][-1]},"
                      f" valid loss: {values['valid']['closs'][-1]},"
                      f" test loss: {values['test']['closs'][-1]}")
                self.best_mcc = np.mean(values['valid']['mcc'][-self.args.n_agg:])
                torch.save(ae.state_dict(), f'{self.complete_log_path}/model_{h}_state.pth')
                torch.save(ae, f'{self.complete_log_path}/model_{h}.pth')
                best_values = get_best_values(values.copy(), ae_only=False, n_agg=self.args.n_agg)
                best_vals = values.copy()
                # best_vals['rec_loss'] = self.best_loss
                # best_vals['dom_loss'] = self.best_dom_loss
                # best_vals['dom_acc'] = self.best_dom_acc
                early_stop_counter = 0
                self.best_iteration = epoch

            if values['valid']['acc'][-1] > self.best_acc:
                print(f"Best Classification Acc Epoch {epoch}, "
                      f"Acc: {values['test']['acc'][-1]}"
                      f"Mcc: {values['test']['mcc'][-1]}"
                      f"Classification train loss: {values['train']['closs'][-1]},"
                      f" valid loss: {values['valid']['closs'][-1]},"
                      f" test loss: {values['test']['closs'][-1]}")

                self.best_acc = values['valid']['acc'][-1]
                early_stop_counter = 0

            if values['valid']['closs'][-1] < self.best_closs:
                print(f"Best Classification Loss Epoch {epoch}, "
                      f"Acc: {values['test']['acc'][-1]} "
                      f"Mcc: {values['test']['mcc'][-1]} "
                      f"Classification train loss: {values['train']['closs'][-1]}, "
                      f"valid loss: {values['valid']['closs'][-1]}, "
                      f"test loss: {values['test']['closs'][-1]}")
                self.best_closs = values['valid']['closs'][-1]
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if self.args.predict_tests and (epoch % 10 == 0):
                loaders = get_loaders_bacteria(all_data, data, self.args.random_recs, self.args.triplet_dloss, ae,
                                               ae.classifier, bs=self.args.bs)
            if self.args.prune_threshold > 0:
                n_neurons = ae.prune_model_paperwise(False, False, weight_threshold=self.args.prune_threshold)
                # If save neptune is True, save the model
                if self.args.log_neptune:
                    log_num_neurons(run, n_neurons, init_n_neurons)

        self.best_mccs += [self.best_mcc]
        self.best_closses += [self.best_closs]

        best_lists, traces = get_empty_traces()
        # Copy list test and make it urinespositives
        best_lists['urinespositives'] = best_lists['test']
        traces['urinespositives'] = traces['test']

        # Verify the model exists
        if not os.path.exists(f'{self.complete_log_path}/model_{h}_state.pth'):
            return -1

        # Loading best model that was saved during training
        ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}_state.pth'))
        # Need another model because the other cant be use to get shap values
        if h == 1:
            shap_ae.load_state_dict(torch.load(f'{self.complete_log_path}/model_{h}_state.pth'))
        # ae.load_state_dict(sd)
        ae.eval()
        shap_ae.eval()
        ae.mapper.eval()
        shap_ae.mapper.eval()
        with torch.no_grad():
            for group in list(data['inputs'].keys()):
                # if group in ['all', 'all_pool']:
                #     continue
                closs, best_lists, traces = self.loop(group, None, ae, sceloss,
                                                      loaders[group], best_lists, traces, nu=0, mapping=False)
        self.log_rep(best_lists, best_vals, best_values, traces, metrics, run, loggers, ae,
                     shap_ae, h, epoch)
        return ae

    def train(self, h_params):
        best_iteration = []
        metrics = {}
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        scaler_name = 'none'
        hparams = {}
        n_aug = 0
        p = 0
        g = 0
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
            'batches_labels': copy.deepcopy(self.data['batches_labels']),
            'batches': copy.deepcopy(self.data['batches']),
            'concs': copy.deepcopy(self.data['concs']),
            'names': copy.deepcopy(self.data['names']),
            # 'sets': copy.deepcopy(self.data['sets']),
            'cats': copy.deepcopy(self.data['cats']),
            'urines': copy.deepcopy(self.data['urines']),
            'manips': copy.deepcopy(self.data['manips'])
        }

        if self.args.binary:
            all_data['labels']['all'] = np.array([
                'blanc' if label == 'blanc' else 'bact' for label in all_data['labels']['all']
            ])
        # self.unique_labels devrait disparaitre et remplace par self.uniques['labels']
        self.unique_labels = np.array(np.unique(all_data['labels']['all']))
        # place blancs at the end
        blanc_class = np.argwhere(self.unique_labels == 'blanc').flatten()[0]
        self.unique_labels = np.concatenate((['blanc'], np.delete(self.unique_labels, blanc_class)))
        self.model_name = f'binary{self.args.binary}_{self.args.model_name}_{self.args.dloss}'
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
            run['warmup_after_warmup'] = self.args.warmup_after_warmup
            run['train_after_warmup'] = self.args.train_after_warmup
            run["hparams"] = hparams
            run["csv_file"] = self.args.csv_file
            run["model_name"] = self.model_name
            run["groupkfold"] = self.args.groupkfold
            run["dataset_name"] = 'MSML-Bacteria'
            run["scaler_name"] = scaler_name
            run["mz_min"] = self.args.min_mz
            run["mz_max"] = self.args.max_mz
            run["rt_min"] = self.args.min_rt
            run["rt_max"] = self.args.max_rt
            run["mz_bin"] = self.args.mz
            run["rt_bin"] = self.args.rt
            run["path"] = self.log_path
            run["concs"] = self.args.concs
            run["binary"] = self.args.binary
            run["spd"] = self.args.spd
            run["ovr"] = self.args.ovr
            run["train_on"] = self.args.train_on
            run["n_features"] = self.args.n_features
            run["total_features"] = all_data['inputs']['all'].shape[1]
            run["ms_level"] = self.args.ms_level
            run["log"] = self.args.log
            run["batches"] = '-'.join(self.uniques['batches'])
            run["context"] = 'train'
            run["remove_bad_samples"] = self.args.remove_bad_samples
            run["colsample_bytree"] = self.args.colsample_bytree
            run["sparse_matrix"] = self.args.sparse_matrix
            run["max_bin"] = self.args.max_bin
            run["device"] = self.args.device

        else:
            run = None

        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        if self.args.log_plots:
            log_ord(all_data, self.uniques, ord_path, scaler_name, run)
        data = copy.deepcopy(all_data)
        metrics = log_fct(data, scaler_name, metrics)
        if self.args.log_plots:
            log_ord(data, self.uniques2, ord_path, f'{scaler_name}_blancs', run)

        all_data, scaler = scale_data(scaler_name, all_data)

        # Basically just takes the samples back into training
        if all_data['inputs']['urinespositives'].shape[0] > 0:
            # Keep only eco
            from collections import Counter
            print(Counter(all_data['labels']['urinespositives']))
            inds_to_keep = np.argwhere(np.isin(all_data['labels']['urinespositives'], ['eco', 'kpn', 'sag'])).flatten()

            for k in all_data.keys():
                if k == 'inputs':
                    pass
                # elif k == 'sets':
                #     all_data[k]['urinespositives'] = ['urinespositives' for _ in range(len(inds_to_keep))]
                else:
                    all_data[k]['urinespositives'] = all_data[k]['urinespositives'][inds_to_keep]
            all_data['inputs']['urinespositives'] =\
                all_data['inputs']['urinespositives'].loc[all_data['names']['urinespositives'], :]
            # all_data['urines']['urinespositives'] = all_data['urines']['urinespositives'][inds_to_keep]
            # all_data['concs']['urinespositives'] = all_data['concs']['urinespositives'][inds_to_keep]
            # all_data['manips']['urinespositives'] = all_data['manips']['urinespositives'][inds_to_keep]
            all_data['meta'] = {'urinespositives': all_data['inputs']['urinespositives'].iloc[:, :2]}
            all_data['cats'] = {'urinespositives': np.array([
                np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['urinespositives']
            ])}
            # all_data['sets'] = {'urinespositives': np.array(['urinespositives' for _ in range(len(inds_to_keep))])}
            print(Counter(all_data['labels']['urinespositives']))
            if self.args.groupkfold:
                b10_inds = np.argwhere(
                    all_data['batches']['urinespositives'] == np.argwhere(
                        self.uniques['batches'] == 'b10'
                    )[0][0]
                ).flatten()
                bpatients = np.argwhere(all_data['batches']['urinespositives'] == np.argwhere(
                        self.uniques['batches'] == 'bpatients'
                )[0][0]).flatten()
                b_inds = np.concatenate((b10_inds, bpatients))
                urines10 = np.array([x.split('_')[-2] for x in all_data['names']['urinespositives'][b_inds]])
                # Make sure the labels are the good ones in the names and inputs labels
                all_data['inputs']['all'] = pd.concat((
                    all_data['inputs']['all'], all_data['inputs']['urinespositives'].iloc[b_inds]
                ))
                all_data['labels']['all'] = np.concatenate((
                    all_data['labels']['all'], all_data['labels']['urinespositives'][b_inds]
                ))
                all_data['batches']['all'] = np.concatenate((
                    all_data['batches']['all'], all_data['batches']['urinespositives'][b_inds]
                ))
                all_data['names']['all'] = np.concatenate((
                    all_data['names']['all'], all_data['names']['urinespositives'][b_inds]
                ))
                all_data['manips']['all'] = np.concatenate((
                    all_data['manips']['all'], all_data['manips']['urinespositives'][b_inds]
                ))
                all_data['urines']['all'] = np.concatenate((
                    all_data['urines']['all'], all_data['urines']['urinespositives'][b_inds]
                ))
                all_data['concs']['all'] = np.concatenate((
                    all_data['concs']['all'], all_data['concs']['urinespositives'][b_inds]
                ))
                all_data['batches_labels']['all'] = np.concatenate((
                    all_data['batches_labels']['all'], all_data['batches_labels']['urinespositives'][b_inds]
                ))
                # all_data['sets']['all'] = np.concatenate((
                #     np.array(['all'] * all_data['inputs']['all'].shape[0]),
                #     np.array(['urinespositives'] * all_data['inputs']['urinespositives'].shape[0])[b_inds]
                # ))
                all_data['meta']['all'] = all_data['inputs']['all'].iloc[:, :2]
                all_data['cats']['all'] = np.array([
                        np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['all']
                ])

                # Remove b10 and bpatients from urinespositives
                all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'].drop(
                    all_data['inputs']['urinespositives'].index[b_inds]
                )
                all_data['meta']['urinespositives'] = all_data['meta']['urinespositives'].drop(
                    all_data['meta']['urinespositives'].index[b_inds]
                )
                all_data['labels']['urinespositives'] = np.delete(all_data['labels']['urinespositives'], b_inds)
                all_data['batches']['urinespositives'] = np.delete(all_data['batches']['urinespositives'], b_inds)
                all_data['batches_labels']['urinespositives'] = np.delete(
                    all_data['batches_labels']['urinespositives'],
                    b_inds
                )
                all_data['names']['urinespositives'] = np.delete(all_data['names']['urinespositives'], b_inds)
                all_data['manips']['urinespositives'] = np.delete(all_data['manips']['urinespositives'], b_inds)
                # all_data['concs']['urinespositives'] = np.delete(all_data['concs']['urinespositives'], b_inds)
                # all_data['sets']['urinespositives'] = np.delete(all_data['sets']['urinespositives'], b_inds)
                all_data['cats']['urinespositives'] = np.delete(all_data['cats']['urinespositives'], b_inds)
                # Remove urines10 from urinespositives because duplicated
                inds_to_keep = np.array([
                    i for i, x in enumerate(all_data['names']['urinespositives']) if x.split('_')[-2] not in urines10
                ])
                all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'].iloc[inds_to_keep]
                all_data['labels']['urinespositives'] = all_data['labels']['urinespositives'][inds_to_keep]
                all_data['batches']['urinespositives'] = all_data['batches']['urinespositives'][inds_to_keep]
                all_data['names']['urinespositives'] = all_data['names']['urinespositives'][inds_to_keep]
                # all_data['concs']['urinespositives'] = all_data['concs']['urinespositives'][inds_to_keep]
                # all_data['sets']['urinespositives'] = all_data['sets']['urinespositives'][inds_to_keep]
                all_data['cats']['urinespositives'] = all_data['cats']['urinespositives'][inds_to_keep]
                all_data['meta']['urinespositives'] = all_data['meta']['urinespositives'].iloc[inds_to_keep]
            else:
                all_data['cats']['all'] = np.array([
                    np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['all']
                ])
                all_data['meta']['all'] = all_data['inputs']['all'].iloc[:, :2]

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

        columns_stats_over0(all_data['inputs']['all'],
                            infos,
                            {'mz': self.args.mz, 'rt': self.args.rt},
                            False)

        # save scaler
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/scaler_{scaler_name}_tmp.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl', 'wb') as f:
            pickle.dump(all_data['inputs']['all'].columns, f)

        print(f'Iteration: {self.iter}')
        # models = []
        h = 0

        h_params['disc_b_warmup'] = 0
        h_params = self.make_params(h_params)
        os.makedirs(self.complete_log_path, exist_ok=True)
        if self.args.groupkfold and not self.args.test:
            self.args.n_repeats = len(np.unique(all_data['batches']['all']))
        elif self.args.test:
            self.args.n_repeats = 1
        while h < self.args.n_repeats:
            print(f'Fold: {h}')

            all_data, self.unique_labels, self.unique_batches = self.get_data(all_data, h)

            lists['inds']['train'] += [all_data['inds']['train']]
            lists['inds']['valid'] += [all_data['inds']['valid']]
            lists['inds']['test'] += [all_data['inds']['test']]

            lists['names']['train'] += [all_data['names']['train']]
            lists['names']['valid'] += [all_data['names']['valid']]
            lists['names']['test'] += [all_data['names']['test']]

            lists['batches']['train'] += [all_data['batches']['train']]
            lists['batches']['valid'] += [all_data['batches']['valid']]
            lists['batches']['test'] += [all_data['batches']['test']]
            lists['unique_batches']['train'] += [list(np.unique(all_data['batches']['train']))]
            lists['unique_batches']['valid'] += [list(np.unique(all_data['batches']['valid']))]
            lists['unique_batches']['test'] += [list(np.unique(all_data['batches']['test']))]

            print(f"Batches. Train: {np.unique(all_data['batches']['train'])},"
                  f"Valid: {np.unique(all_data['batches']['valid'])},"
                  f"Test: {np.unique(all_data['batches']['test'])}")
            if n_aug > 0:
                all_data['inputs']['train'] = all_data['inputs']['train'].fillna(0)
                all_data['inputs']['train'] = augment_data(all_data['inputs']['train'], n_aug, p, g)
                all_data['labels']['train'] = np.concatenate([all_data['labels']['train']] * (n_aug + 1))
                all_data['batches']['train'] = np.concatenate([all_data['batches']['train']] * (n_aug + 1))
                all_data['names']['train'] = np.concatenate([all_data['names']['train']] * (n_aug + 1))
                all_data['manips']['train'] = np.concatenate([all_data['manips']['train']] * (n_aug + 1))
                # all_data['sets']['train'] = np.concatenate([all_data['sets']['train']] * (n_aug + 1))
                all_data['cats']['train'] = np.concatenate([all_data['cats']['train']] * (n_aug + 1))
            else:
                all_data['inputs']['train'] = all_data['inputs']['train'].fillna(0)
            all_data['inputs']['valid'] = all_data['inputs']['valid'].fillna(0)
            all_data['inputs']['test'] = all_data['inputs']['test'].fillna(0)

            lists['classes']['train'] += [np.array([
                np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['train']
            ])]
            lists['classes']['valid'] += [np.array([
                np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['valid']
            ])]
            lists['classes']['test'] += [np.array([
                np.argwhere(label == self.unique_labels)[0][0] for label in all_data['labels']['test']
            ])]
            lists['labels']['train'] += [all_data['labels']['train']]
            lists['labels']['valid'] += [all_data['labels']['valid']]
            lists['labels']['test'] += [all_data['labels']['test']]
            lists['manips']['train'] += [all_data['manips']['train']]
            lists['manips']['valid'] += [all_data['manips']['valid']]
            lists['manips']['test'] += [all_data['manips']['test']]
            lists['urines']['train'] += [all_data['urines']['train']]
            lists['urines']['valid'] += [all_data['urines']['valid']]
            lists['urines']['test'] += [all_data['urines']['test']]
            lists['concs']['train'] += [all_data['concs']['train']]
            lists['concs']['valid'] += [all_data['concs']['valid']]
            lists['concs']['test'] += [all_data['concs']['test']]

            self.data['inputs']['train'] = all_data['inputs']['train']
            self.data['labels']['train'] = all_data['labels']['train']
            self.data['batches']['train'] = all_data['batches']['train']
            self.data['names']['train'] = all_data['names']['train']
            self.data['inputs']['valid'] = all_data['inputs']['valid']
            self.data['labels']['valid'] = all_data['labels']['valid']
            self.data['batches']['valid'] = all_data['batches']['valid']
            self.data['names']['valid'] = all_data['names']['valid']
            self.data['inputs']['test'] = all_data['inputs']['test']
            self.data['labels']['test'] = all_data['labels']['test']
            self.data['batches']['test'] = all_data['batches']['test']
            self.data['names']['test'] = all_data['names']['test']
            self.data['cats']['train'] = all_data['cats']['train']
            self.data['cats']['valid'] = all_data['cats']['valid']
            self.data['cats']['test'] = all_data['cats']['test']
            self.data['meta'] = {}
            self.data['meta']['all'] = all_data['meta']['all']
            self.data['meta']['urinespositives'] = all_data['meta']['urinespositives']
            self.data['meta']['train'] = all_data['meta']['train']
            self.data['meta']['valid'] = all_data['meta']['valid']
            self.data['meta']['test'] = all_data['meta']['test']
            # self.data['sets']['train'] = all_data['sets']['train']
            # self.data['sets']['valid'] = all_data['sets']['valid']
            # self.data['sets']['test'] = all_data['sets']['test']
            self.data['manips']['train'] = all_data['manips']['train']
            self.data['manips']['valid'] = all_data['manips']['valid']
            self.data['manips']['test'] = all_data['manips']['test']
            self.data['urines']['train'] = all_data['urines']['train']
            self.data['urines']['valid'] = all_data['urines']['valid']
            self.data['urines']['test'] = all_data['urines']['test']
            self.data['concs']['train'] = all_data['concs']['train']
            self.data['concs']['valid'] = all_data['concs']['valid']
            self.data['concs']['test'] = all_data['concs']['test']
            # self.data['urines']['train'] = all_data['urines']['train']
            # self.data['urines']['valid'] = all_data['urines']['valid']
            # self.data['urines']['test'] = all_data['urines']['test']

            m = self.train_bernn(all_data, h, h_params, run)

            try:
                assert m is not int
            except Exception as e:
                print('m is int', e)

            self.dump_model(h, m, scaler_name, lists)
            best_iteration += [self.best_iteration]
            train_preds = m.predict(
                torch.Tensor(all_data['inputs']['train'].to_numpy()).to(self.args.device)
            )
            valid_preds = m.predict(
                torch.Tensor(all_data['inputs']['valid'].to_numpy()).to(self.args.device)
            )
            test_preds = m.predict(
                torch.Tensor(all_data['inputs']['test'].to_numpy()).to(self.args.device)
            )
            lists['acc']['train'] += [ACC(train_preds, lists['classes']['train'][-1])]
            lists['preds']['train'] += [train_preds]

            lists['acc']['valid'] += [ACC(valid_preds, lists['classes']['valid'][-1])]
            lists['acc']['test'] += [ACC(test_preds, lists['classes']['test'][-1])]
            lists['preds']['valid'] += [valid_preds]
            lists['preds']['test'] += [test_preds]
            lists['mcc']['train'] += [MCC(lists['classes']['train'][-1], lists['preds']['train'][-1])]
            lists['mcc']['valid'] += [MCC(lists['classes']['valid'][-1], lists['preds']['valid'][-1])]
            lists['mcc']['test'] += [MCC(lists['classes']['test'][-1], lists['preds']['test'][-1])]

            if all_data['inputs']['urinespositives'].shape[0] > 0:
                lists['preds']['posurines'] += [
                    m.predict(torch.Tensor(all_data['inputs']['urinespositives'].to_numpy()).to(self.args.device))
                ]
                lists['labels']['posurines'] += [all_data['labels']['urinespositives']]
                lists['classes']['posurines'] += [
                    np.array([
                        np.argwhere(
                            label == self.unique_labels
                        )[0][0] for label in all_data['labels']['urinespositives']
                    ])
                ]
                lists['names']['posurines'] += [all_data['names']['urinespositives']]
                lists['batches']['posurines'] += [all_data['batches']['urinespositives']]
                lists['manips']['posurines'] += [all_data['manips']['urinespositives']]
                try:
                    lists['proba']['posurines'] += [
                        m.predict_proba(
                            torch.Tensor(all_data['inputs']['urinespositives'].to_numpy()).to(self.args.device)
                        )
                    ]
                except Exception as e:
                    print('bernn proba error train', e)
                    pass

            try:
                lists['proba']['train'] += [m.predict_proba(
                    torch.Tensor(all_data['inputs']['train'].to_numpy()).to(self.args.device)
                )]
                lists['proba']['valid'] += [m.predict_proba(
                    torch.Tensor(all_data['inputs']['valid'].to_numpy()).to(self.args.device)
                )]
                lists['proba']['test'] += [m.predict_proba(
                    torch.Tensor(all_data['inputs']['test'].to_numpy()).to(self.args.device)
                )]
                data_list = {
                    'valid': {
                        'inputs': all_data['inputs']['valid'],
                        'labels': lists['classes']['valid'][-1],
                        'preds': lists['preds']['valid'][-1],
                        'proba': lists['proba']['valid'][-1],
                        'batches': all_data['batches']['valid'],
                        'names': lists['proba']['valid'][-1]
                    },
                    'test': {
                        'inputs': all_data['inputs']['test'],
                        'labels': lists['classes']['test'][-1],
                        'preds': lists['preds']['test'][-1],
                        'proba': lists['proba']['test'][-1],
                        'batches': all_data['batches']['test'],
                        'names': lists['proba']['test'][-1]
                    }
                }
            except Exception as e:
                print('bernn predict_proba error train', e)
                data_list = {
                    'valid': {
                        'inputs': all_data['inputs']['valid'],
                        'labels': lists['classes']['valid'][-1],
                        'preds': lists['preds']['valid'][-1],
                        'batches': all_data['batches']['valid'],
                        'names': lists['names']['valid'][-1],
                        'proba': lists['proba']['valid'][-1]
                    },
                    'test': {
                        'inputs': all_data['inputs']['test'],
                        'labels': lists['classes']['test'][-1],
                        'preds': lists['preds']['test'][-1],
                        'batches': all_data['batches']['test'],
                        'names': lists['names']['test'][-1],
                        'proba': lists['proba']['test'][-1]
                    }
                }

            if self.best_scores['acc']['valid'] is None:
                self.best_scores['acc']['valid'] = 0

            h += 1
            try:
                np.concatenate(lists['proba']['posurines']).max(1)
            except Exception as e:
                print('Error at concat of probas', e)

        try:
            np.concatenate(lists['proba']['posurines']).max(1)
        except Exception as e:
            print('Error at concat of probas after repeat loop', e)
        if self.args.groupkfold:
            batches = np.concatenate([np.unique(x) for x in lists['batches']['valid']])
            toprint = [f"{batches[i]}:{lists['mcc']['valid'][i]}" for i in range(len(batches))]
            print(toprint)
        else:
            print(lists['mcc']['valid'])
        print('valid_acc:', np.mean(lists['acc']['valid']),
              'valid_mcc:', np.mean(lists['mcc']['valid']),
              'scaler:', scaler_name,
              'h_params:', param_grid
              )
        posurines_df = None  # TODO move somewhere more logical

        # Log in neptune the optimal iteration
        if self.args.log_neptune:
            run["best_iteration"] = np.round(np.mean([x for x in best_iteration]))
            run["model_size"] = get_size_in_mb(self.ae)

        try:
            np.concatenate(lists['proba']['posurines']).max(1)
        except Exception as e:
            print('Error at concat of probas after repeat loop 2', e)
        lists, posurines_df = self.save_confusion_matrices(lists, run)
        self.save_calibration_curves(lists, run)
        self.save_roc_curves(lists, run)
        if np.mean(lists['mcc']['valid']) > np.mean(self.best_scores['mcc']['valid']):
            if self.args.log_shap:
                Xs = {
                    'train': all_data['inputs']['train'],
                    'valid': all_data['inputs']['valid'],
                    'test': all_data['inputs']['test'],
                    # 'posurines': all_data['inputs']['urinespositives'],
                }
                ys = {
                    'train': lists['classes']['train'][-1],
                    'valid': lists['classes']['valid'][-1],
                    'test': lists['classes']['test'][-1],
                    # 'posurines': np.array([
                    #    np.argwhere(l == self.unique_labels)[0][0] for l in all_data['labels']['urinespositives']
                    #  ]),
                }
                labels = {
                    'train': lists['labels']['train'][-1],
                    'valid': lists['labels']['valid'][-1],
                    'test': lists['labels']['test'][-1],
                    # 'posurines': np.array([
                    #   np.argwhere(l == self.unique_labels)[0][0] for l in all_data['labels']['urinespositives']
                    #  ]),
                }
                args_dict = {
                    'inputs': Xs,
                    'ys': ys,
                    'labels': labels,
                    'model': m,
                    'model_name': self.args.model_name,
                    'log_path': self.log_path,
                }
                run = log_shap(run, args_dict)
                run['log_shap'] = 1
            else:
                run['log_shap'] = 0

            # save the features kept
            with open(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl', 'wb') as f:
                pickle.dump(data_list['test']['inputs'].columns, f)

            if self.keep_models:
                self.save_models(scaler_name)
            # Save the individual scores of each sample with class, #batch
            self.save_results_df(lists, run)
            self.retrieve_best_scores(lists)
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
            run['posurines/individual_results'].upload(
                f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
            )

        if self.args.log_neptune:
            log_neptune(run, lists, best_scores)
            run.stop()
            # model.stop()

        return 1 - np.mean(lists['mcc']['valid'])

    def get_ordered_layers(self, params):
        """Extract layer parameters from params dictionary, order them, and store in a new dictionary.

        Args:
            params (dict): Dictionary containing model parameters including layer sizes

        Returns:
            dict: Ordered dictionary of layer parameters
        """
        # Extract layer parameters and sort them
        layer_params = {k: v for k, v in params.items() if k.startswith('layer')}
        ordered_layers = dict(sorted(layer_params.items(),
                                     key=lambda x: int(x[0].replace('layer', ''))))
        return ordered_layers

    def _convert_to_serializable(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

    def add_encoder_decoder_outputs(self, all_data, lists):
        """Add encoder and decoder outputs from lists to all_data.

        Args:
            all_data (dict): Dictionary containing the data
            lists (dict): Dictionary containing model outputs including encoded_values and dec

        Returns:
            dict: Updated all_data with encoder and decoder outputs
        """
        if 'encoded_values' in lists and 'rec_values' in lists:
            # Add encoder outputs
            all_data['encoded_values'] = {
                'train': lists['encoded_values']['train'],
                'valid': lists['encoded_values']['valid'],
                'test': lists['encoded_values']['test'],
                'all': np.concatenate([
                    lists['encoded_values']['train'],
                    lists['encoded_values']['valid'],
                    lists['encoded_values']['test']
                ])
            }

            # Add decoder outputs
            all_data['rec_values'] = {
                'train': lists['rec_values']['train'],
                'valid': lists['rec_values']['valid'],
                'test': lists['rec_values']['test'],
                'all': np.concatenate([
                    lists['rec_values']['train'],
                    lists['rec_values']['valid'],
                    lists['rec_values']['test']
                ])
            }

            # Add urinespositives if they exist
            if 'urinespositives' in lists['encoded_values']:
                all_data['encoded_values']['urinespositives'] = lists['encoded_values']['urinespositives']
            if 'urinespositives' in lists['rec_values']:
                all_data['rec_values']['urinespositives'] = lists['rec_values']['urinespositives']

        return all_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Add relevant arguments here, e.g. for data, model, etc.
    parser.add_argument('--device', type=str, default='cuda:0')
    # ... (add more as needed)
    args = parser.parse_args()

    # TODO: Adapt data loading and preprocessing for spectra tokenization
    # TODO: Adapt training loop for LargeSpectralBERT
    train = TrainLSM(args, path=None)  # Add other required args as needed
    train.load_autoencoder()
    # train.train()  # Uncomment and adapt when ready
