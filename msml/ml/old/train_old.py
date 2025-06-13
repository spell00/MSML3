import copy
import json
import os
import neptune
import pickle
import numpy as np
import pandas as pd
from utils import remove_zero_cols, scale_data, augment_data, get_empty_lists
from loggings import log_ord, log_fct, save_confusion_matrix, log_neptune, plot_bars
from sklearn_train_nocv import get_confusion_matrix, plot_roc
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
from scipy import stats
from log_shap import log_shap
import matplotlib.pyplot as plt
from utils import columns_stats_over0
import sys


NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
# NEPTUNE_PROJECT_NAME = "Bacteria-MSMS"
# NEPTUNE_MODEL_NAME = 'BAC-'
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'


def get_size_in_mb(obj):
    size_in_bytes = sys.getsizeof(obj)
    size_in_mb = size_in_bytes / (1024 * 1024)
    return size_in_mb


class Train:
    def __init__(self, name, model, data, uniques, hparams_names,
                 log_path, args, logger, log_neptune, mlops='None',
                 binary_path=None):
        self.log_neptune = log_neptune
        self.best_roc_score = -1
        self.args = args
        self.log_path = log_path
        self.binary_path = binary_path
        self.model = model
        self.data = data
        self.logger = logger
        self.hparams_names = hparams_names
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
        self.mlops = mlops
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
            'concs': copy.deepcopy(self.data['concs']),
            'names': copy.deepcopy(self.data['names']),
        }

        if self.args.binary:
            all_data['labels']['all'] = np.array(['blanc' if label =='blanc' else 'bact' for label in all_data['labels']['all']])
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

        if self.log_neptune:
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

            lists['classes']['train'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in train_labels])]
            lists['classes']['valid'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in valid_labels])]
            lists['classes']['test'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in test_labels])]
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
                
            if self.best_scores['acc']['valid'] is None:
                self.best_scores['acc']['valid'] = 0

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
        if self.log_neptune:
            model["best_iteration"] = run["best_iteration"] = np.round(np.mean([x for x in best_iteration]))
            model["model_size"] = run["model_size"] = get_size_in_mb(m)

        lists, posurines_df = self.save_confusion_matrices(all_data, lists, run)
        if np.mean(lists['mcc']['valid']) > np.mean(self.best_scores['mcc']['valid']):
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
                    # 'posurines': np.array([np.argwhere(l == self.unique_labels)[0][0] for l in all_data['labels']['urinespositives']]),
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
            
            self.keep_models(scaler_name)
            # Save the individual scores of each sample with class, #batch
            self.save_results_df(lists, run)
            self.retrieve_best_scores(lists)
            best_scores = self.save_best_model_hparams(param_grid, other_params, scaler_name, lists['unique_batches'], metrics)
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

        if self.log_neptune:
            log_neptune(run, lists, best_scores)
            run.stop()
            model.stop()

        return 1 - np.mean(lists['mcc']['valid'])

    def train_no_split(self, h_params):
        metrics = {}
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        scaler_name = 'none'
        hparams = {}
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
            'concs': copy.deepcopy(self.data['concs']),
            'names': copy.deepcopy(self.data['names']),
        }

        if self.args.binary:
            all_data['labels']['all'] = np.array(['blanc' if label=='blanc' else 'bact' for label in all_data['labels']['all']])
        # self.unique_labels devrait disparaitre et remplace par self.uniques['labels']
        self.unique_labels = np.array(np.unique(all_data['labels']['all']))
        # place blancs at the end
        blanc_class = np.argwhere(self.unique_labels == 'blanc').flatten()[0]
        self.unique_labels = np.concatenate((['blanc'], np.delete(self.unique_labels, blanc_class)))
        self.model_name = f'binary{self.args.binary}_{self.args.model_name}'
        self.uniques['labels'] = self.unique_labels

        not_zeros_col = remove_zero_cols(all_data['inputs']['all'], threshold)

        all_data['inputs']['all'] = all_data['inputs']['all'].iloc[:, not_zeros_col]
        all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'].iloc[:, not_zeros_col]

        if self.log_neptune:
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
            model['context'] = run['context'] = 'inference'
            model['remove_bad_samples'] = run['remove_bad_samples'] = self.args.remove_bad_samples

        else:
            model = None
            run = None

        all_data, scaler = scale_data(scaler_name, all_data)

        # save scaler
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/scaler_{scaler_name}_tmp.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl', 'wb') as f:
            pickle.dump(all_data['inputs']['all'].columns, f)

        print(f'Iteration: {self.iter}')

        if self.args.groupkfold:
            self.args.n_repeats = len(np.unique(all_data['batches']['all']))
                            
        lists['names']['posurines'] = np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])
        lists['batches']['posurines'] = all_data['batches']['urinespositives']

        lists['names']['train'] = all_data['names']['all']

        lists['batches']['train'] = all_data['batches']['all']
        lists['unique_batches']['train'] = list(np.unique(all_data['batches']['all']))
        lists['labels']['train'] = all_data['labels']['all']

        if n_aug > 0:
            train_data = augment_data(all_data['inputs']['all'], n_aug, p, g)
            train_data = np.nan_to_num(train_data)
            train_labels = np.concatenate([lists['labels']['train']] * (n_aug + 1))
            train_batches = np.concatenate([lists['batches']['train']] * (n_aug + 1))
        else:
            train_data = all_data['inputs']['all'].fillna(0)
            train_labels = lists['labels']['train']
            train_batches = lists['batches']['train']

        lists['classes']['train'] = np.array([np.argwhere(l == self.unique_labels)[0][0] for l in train_labels])
        lists['labels']['train'] = train_labels

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

        m.fit(train_data, lists['classes']['train'], verbose=True)
        try:
            lists['acc']['train'] = m.score(train_data, lists['classes']['train'])
            lists['preds']['train'] = m.predict(train_data)
        except:
            lists['acc']['train'] = m.score(train_data.values, lists['classes']['train'])
            lists['preds']['train'] = m.predict(train_data.values)

        if all_data['inputs']['urinespositives'].shape[0] > 0:
            try:
                lists['preds']['posurines'] = m.predict(all_data['inputs']['urinespositives'])
                try:
                    lists['proba']['posurines'] = m.predict_proba(all_data['inputs']['urinespositives'])
                except:
                    pass

            except:
                lists['preds']['posurines'] = m.predict(all_data['inputs']['urinespositives'].values)
                try:
                    lists['proba']['posurines'] = m.predict_proba(all_data['inputs']['urinespositives'].values)
                except:
                    pass

        try:
            lists['proba']['train'] = m.predict_proba(train_data)
        except:
            pass
        lists['mcc']['train'] = MCC(lists['classes']['train'], lists['preds']['train'])
        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        log_ord(self.data, self.uniques, ord_path, scaler_name, run)
        data = copy.deepcopy(self.data)
        metrics = log_fct(data, scaler_name, metrics)
        log_ord(data, self.uniques2, ord_path, f'{scaler_name}_blancs', run)
        self.dump_model(0, m, scaler_name, None)
        self.keep_models(scaler_name)

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

        with open(f'{self.log_path}/saved_models/unique_batches_{self.name}_{self.args.model_name}.json', "w") as read_file:
            json.dump(unique_batches, read_file)

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
        
        self.best_params_dict['ami'] = metrics[scaler_name]['all']['adjusted_mutual_info_score']['domains']
        self.best_params_dict['ari'] = metrics[scaler_name]['all']['adjusted_rand_score']['domains']
        n_batches = np.unique(
            np.concatenate([
                np.concatenate(unique_batches[b]) for b in unique_batches if len(unique_batches[b]) > 0
            ])
        ).flatten().shape[0]
        self.best_params_dict['nBE'] = (np.log(n_batches) - metrics[scaler_name]['all']['shannon']['domains'])  / np.log(n_batches)

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

        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}.json', "w") as read_file:
            json.dump(self.best_params_dict, read_file)
        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}_values.json', "w") as read_file:
            json.dump(self.best_params_dict_values, read_file)
        # add ari, ami and nbe to run
        return self.best_params_dict

    def make_predictions(self, all_data, lists, run):
        # TODO Make urinespositives_real_df complete by adding the missing blank samples
        urinespositives_names = np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])
        urinespositives_batches = np.array([x for x in all_data['batches']['urinespositives']])
        # TODO load expected classes elsewhere
        urinespositives_real_df = pd.concat((
            pd.read_csv(f'resources/bacteries_2024/B10-05-03-2024/b10_patients_samples.csv'),
            pd.read_csv(f'resources/bacteries_2024/B11-05-24-2024/b11_patients_samples.csv')
        ))

        urinespositives_names = np.array([x for x in urinespositives_names if x in urinespositives_real_df.loc[:, 'ID'].to_numpy()])
        urinespositives_names = np.unique(urinespositives_names)
        urinespositives_real_df.loc[:, 'Class'] = [l.lower() for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]
        if self.args.binary:
            urinespositives_real_df.loc[:, 'Class'] = ['blanc' if l == 'blanc' else 'bact' for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]
        intersect = np.intersect1d(urinespositives_names, urinespositives_real_df.loc[:, 'ID'].to_numpy())
        to_keep = [i for i, x in enumerate(urinespositives_names) if x in intersect]

        urinespositives_names = urinespositives_names[to_keep]
        urinespositives_batches = urinespositives_batches[to_keep]

        to_keep = [i for i, x in enumerate(urinespositives_real_df.loc[:, 'ID']) if x in intersect]
        urinespositives_real_df = urinespositives_real_df.iloc[to_keep]

        new_order = np.argsort(urinespositives_names)
        urinespositives_real_df = urinespositives_real_df.iloc[new_order]

        blanc_ids = [
            np.array([i for i, x in enumerate(lists['labels']['valid'][j]) if x == 'blanc']) for j in range(len(lists['labels']['valid']))
        ]
        # lists['names']['posurines'] = [
        #     [lists['names']['posurines'][j][i] for i in to_keep] + [lists['names']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['names']['posurines']))
        # ]
        # lists['batches']['posurines'] = [
        #     [lists['batches']['posurines'][j][i] for i in to_keep] + [lists['batches']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['batches']['posurines']))
        # ]
        # Get the labels of urinespositives_real_df in the same order as the names
        # lists['labels']['posurines'] = [
        #     urinespositives_real_df.loc[:, 'Class'].to_numpy().tolist() + [lists['labels']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['preds']['posurines']))
        # ]

        lists['names']['posurines'] = [
            [lists['names']['posurines'][j][i] for i in to_keep] for j in range(len(lists['names']['posurines']))
        ]
        lists['batches']['posurines'] = [
            [lists['batches']['posurines'][j][i] for i in to_keep] for j in range(len(lists['batches']['posurines']))
        ]
        # Get the labels of urinespositives_real_df in the same order as the names
        lists['labels']['posurines'] = [
            urinespositives_real_df.loc[:, 'Class'].to_numpy().tolist() for j in range(len(lists['preds']['posurines']))
        ]

        # urinespositives_real_df = urinespositives_real_df.loc[:, 'Class'].to_numpy()
        
        try:
            lists['preds']['posurines'] = [
                [lists['preds']['posurines'][j][i] for i in to_keep] for j in range(len(lists['preds']['posurines']))
            ]
            lists['proba']['posurines'] = [
                [lists['proba']['posurines'][j][i] for i in to_keep] for j in range(len(lists['proba']['posurines']))
            ]
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

        except:
            # lists['preds']['posurines'] = [
            #     [lists['preds']['posurines'][j][i] for i in to_keep] + [lists['preds']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['preds']['posurines']))
            # ]
            lists['preds']['posurines'] = [
                [lists['preds']['posurines'][j][i] for i in to_keep] for j in range(len(lists['preds']['posurines']))
            ]
            posurines_df = pd.DataFrame(
                {
                    'names': np.concatenate([urinespositives_names for _ in range(len(lists['preds']['posurines']))]),
                    'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['posurines'])]),
                    'batches': np.concatenate([urinespositives_batches for _ in range(len(lists['preds']['posurines']))]),
                    'preds': np.concatenate(lists['preds']['posurines']),
                    'labels': np.concatenate([urinespositives_real_df.loc[:, 'Class'] for _ in range(len(lists['preds']['posurines']))]),                
                }
            )
        posurines_df.loc[:, 'preds'] = [
            self.unique_labels[l] for l in posurines_df.loc[:, 'preds'].to_numpy()
        ]
        # assert 'bact' not in posurines_df.loc[:, 'preds']
        posurines_df.to_csv(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
        )
        try:
            lists['proba']['posurines'] = np.stack(lists['proba']['posurines'])
        except:
            return posurines_df, lists
        def flatten(xss):
            return [x for xs in xss for x in xs]        
        blk_names = flatten([
            [lists['names']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['names']['posurines']))
        ])
        blk_batches = flatten([
            [lists['batches']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['batches']['posurines']))
        ])
        blk_labels = flatten([
            [lists['labels']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['labels']['posurines']))
        ])
        blk_preds = flatten([
            [lists['preds']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['preds']['posurines']))
        ])
        try:
            proba_posurines = np.mean(lists['proba']['posurines'], 0)
            preds_posurines = proba_posurines.argmax(1)
            best_proba_posurines = np.array([
                proba_posurines[i, x] for i, x in enumerate(preds_posurines)
            ])
            blk_proba = np.stack(flatten([
                [lists['proba']['valid'][j][i] for i in blanc_ids[j]] for j in range(len(lists['proba']['posurines']))
            ]))

            posurines_df = pd.DataFrame(
                {
                    'names': lists['names']['posurines'][0] + blk_names,
                    'batches': lists['batches']['posurines'][0] + blk_batches,
                    'preds': lists['preds']['posurines'][0] + blk_preds,
                    'labels': lists['labels']['posurines'][0] + blk_labels,                
                    'proba': np.concatenate((proba_posurines.max(1).flatten(), blk_proba.max(1).flatten())),                    
                }
            )
            proba_posurines = np.concatenate((proba_posurines, blk_proba))
            for i, label in enumerate(self.unique_labels):
                posurines_df[label] = proba_posurines[:, i]
        except:
            preds_posurines = np.stack(lists['preds']['posurines'])
            preds_posurines = stats.mode(preds_posurines, axis=0)[0].flatten()
            posurines_df = pd.DataFrame(
                {
                    'names': urinespositives_names,
                    'batches': urinespositives_batches,
                    'preds': preds_posurines,
                    'labels': urinespositives_real_df.loc[:, 'Class'],                
                }
            )

        posurines_df.loc[:, 'preds'] = [
            self.unique_labels[l] for l in posurines_df.loc[:, 'preds'].to_numpy()
        ]
        assert 'bact' not in posurines_df.loc[:, 'preds']
        posurines_df.to_csv(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_mode_results.csv'
        )
        run[f'posurines/individual_results'].upload(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
        )
        run[f'posurines/mode_results'].upload(
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
            self.unique_labels[l] for l in df_valid.loc[:, 'preds'].to_numpy()
        ]
        df_test.loc[:, 'preds'] = [
            self.unique_labels[l] for l in df_test.loc[:, 'preds'].to_numpy()
        ]

        df_valid.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_valid_individual_results.csv')
        df_test.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_test_individual_results.csv')
        run[f'valid/individual_results'].upload(f'{self.log_path}/saved_models/{self.args.model_name}_valid_individual_results.csv')
        run[f'test/individual_results'].upload(f'{self.log_path}/saved_models/{self.args.model_name}_test_individual_results.csv')
        self.save_thresholds_curve('valid', lists, run)
        self.save_thresholds_curve('test', lists, run)
        self.save_thresholds_curve0('valid', df_valid, run)
        self.save_thresholds_curve0('test', df_test, run)
        # Do the same but with posurines, if any
            
        plot_bars(self.args, run, self.unique_labels)
        
    def save_roc_curves(self, lists, run):
        try:
            self.best_roc_train = plot_roc(lists['proba']['train'], lists['classes']['train'], self.unique_labels,
                            f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_train", 
                            binary=self.args.binary, acc=lists['acc']['train'], run=run)
        except:
            pass
        try:
            self.best_roc_valid = plot_roc(lists['proba']['valid'], lists['classes']['valid'], self.unique_labels,
                            f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_valid", 
                            binary=self.args.binary, acc=lists['acc']['valid'], run=run)
        except:
            pass
        try:
            self.best_roc_test = plot_roc(lists['proba']['test'], lists['classes']['test'], self.unique_labels,
                            f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_test",
                            binary=self.args.binary, acc=lists['acc']['test'], run=run)
        except:
            pass

    def save_thresholds_curve0(self, group, df, run):
        accs = []
        mccs = []
        proportion_predicted = []
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thres in thresholds:
            df1 = df.copy()
            inds = [i for i, proba in enumerate(df.loc[:, 'proba'].to_numpy()) if proba > thres ]
            df1 = df1.iloc[inds]
            df1.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_{thres}.csv')
            run[f'{group}/individual_results_{thres}'].upload(f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_{thres}.csv')
            accs += [ACC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
            mccs += [MCC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
            proportion_predicted += [df1.shape[0] / df.shape[0]]
        fig = plt.figure()
        plt.plot(thresholds, accs, label='acc')
        plt.plot(thresholds, mccs, label='mcc')
        plt.plot(thresholds, proportion_predicted, label='FOT') # Fraction > threshold
        # Add a dash line at 0.95 on y
        plt.axhline(0.95, color='black', linestyle='--')
        plt.legend()
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        # plt.xlim(0, 9)
        fig.savefig(f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_thresholds.png')
        run[f'{group}/thresholds'].upload(f'{self.log_path}/ROC/{self.name}_{self.args.model_name}_{group}_thresholds.png')
        
    def save_thresholds_curve(self, group, lists, run):
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        accs = {x: [] for x in thresholds}
        mccs = {x: [] for x in thresholds}
        proportion_predicted = {x: [] for x in thresholds}
        for batch in range(len(lists['proba'][group])):
            try:
                probs = lists['proba'][group][batch].max(1)
            except:
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
            # for i, label in enumerate(self.unique_labels):
            #     df[label] = lists['proba'][group][batch][:, i]
            df.loc[:, 'preds'] = [
                self.unique_labels[l] for l in df.loc[:, 'preds'].to_numpy()
            ]

            if len(np.unique(df.loc[:, 'batches'].to_numpy())) == 1:
                batch_name = df.loc[:, 'batches'].to_numpy()[0]
            else:
                batch_name = batch
            for thres in thresholds:
                df1 = df.copy()
                inds = [i for i, proba in enumerate(df.loc[:, 'proba'].to_numpy()) if proba > thres ]
                df1 = df1.iloc[inds]
                df1.to_csv(
                    f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_{thres}_{batch_name}.csv'
                )
                run[f'{group}/individual_results_{thres}_{batch_name}'].upload(
                    f'{self.log_path}/saved_models/{self.args.model_name}_{group}_individual_results_{thres}_{batch_name}.csv'
                    )
                accs[thres] += [ACC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
                mccs[thres] += [MCC(df1.loc[:, 'preds'].to_numpy(), df1.loc[:, 'labels'].to_numpy())]
                proportion_predicted[thres] += [df1.shape[0] / df.shape[0]]
        fig = plt.figure()
        plt.axhline(0.95, color='black', linestyle='--')
        plt.plot(thresholds, [np.mean(accs[k]) for k in accs.keys()], label='acc', color='blue')
        plt.plot(thresholds, [np.mean(mccs[k]) for k in mccs.keys()], label='mcc', color='red')
        plt.plot(thresholds, [np.mean(proportion_predicted[k]) for k in proportion_predicted.keys()], label='FOT', color='black') # Fraction > threshold
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
                f.write(f'{thres},{np.mean(accs[thres])},{np.std(accs[thres])},{np.mean(mccs[thres])},{np.std(mccs[thres])},{np.mean(proportion_predicted[thres])},{np.std(proportion_predicted[thres])}\n')
        run[f'{group}/thresholds_table'].upload(
            f'{self.log_path}/saved_models/table_{self.args.model_name}_{group}_thresholds.csv'
        )
    def keep_models(self, scaler_name):
        """
        Remove the tmp from the name if the models are to be kept because the best yet
        """
        for f in os.listdir(f'{self.log_path}/saved_models/'):
            if f.startswith(f'{self.args.model_name}_{scaler_name}') and f.endswith('tmp.pkl'):
                os.rename(f'{self.log_path}/saved_models/{f}', f'{self.log_path}/saved_models/{f[:-8]}.pkl')

        os.rename(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl', f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}.pkl')
        os.rename(f'{self.log_path}/saved_models/scaler_{scaler_name}_tmp.pkl', f'{self.log_path}/saved_models/scaler_{scaler_name}.pkl')

    def remove_models(self, scaler_name):
        """
        Remove all models saved with the given scaler_name
        """
        for f in os.listdir(f'{self.log_path}/saved_models/'):
            if f.startswith(f'{self.args.model_name}_{scaler_name}') and f.endswith('tmp.pkl'):
                os.remove(f'{self.log_path}/saved_models/{f}')
        os.remove(f'{self.log_path}/saved_models/scaler_{scaler_name}_tmp.pkl')
        os.remove(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl')

    def dump_models(self, models, scaler_name, lists):
        # Save unique labels
        with open(f'{self.log_path}/saved_models/unique_labels.json', "w") as read_file:
            json.dump(self.unique_labels.tolist(), read_file)

        for i, m in enumerate(models):
            # save model
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}.pkl', 'wb') as f:
                pickle.dump(m, f)
            # save indices
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_train_indices.pkl', 'wb') as f:
                pickle.dump(lists['inds']['train'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_valid_indices.pkl', 'wb') as f:
                pickle.dump(lists['inds']['valid'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_test_indices.pkl', 'wb') as f:
                pickle.dump(lists['inds']['test'][i], f)

    def dump_model(self, i, m, scaler_name, lists):
        # Save unique labels
        with open(f'{self.log_path}/saved_models/unique_labels.json', "w") as read_file:
            json.dump(self.unique_labels.tolist(), read_file)

        # save model
        with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_tmp.pkl', 'wb') as f:
            pickle.dump(m, f)

        if lists is not None:
            # save indices
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_train_indices_tmp.pkl', 'wb') as f:
                pickle.dump(lists['inds']['train'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_valid_indices_tmp.pkl', 'wb') as f:
                pickle.dump(lists['inds']['valid'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{scaler_name}_{i}_test_indices_tmp.pkl', 'wb') as f:
                pickle.dump(lists['inds']['test'][i], f)

    def retrieve_best_scores(self, lists):
        self.best_scores['acc']['train'] = lists['acc']['train']
        self.best_scores['acc']['valid'] = lists['acc']['valid']
        self.best_scores['acc']['test'] = lists['acc']['test']
        self.best_scores['mcc']['train'] = lists['mcc']['train']
        self.best_scores['mcc']['valid'] = lists['mcc']['valid']
        self.best_scores['mcc']['test'] = lists['mcc']['test']
        
    def save_confusion_matrices(self, all_data, lists, run):
        relevant_samples = []
        posurines_df, lists = self.make_predictions(all_data, lists, run)
        relevant_samples = [i for i, l in enumerate(posurines_df.loc[:, 'labels'].to_numpy()) if l in self.unique_labels]
        if len(relevant_samples) > 0:
            posurines_df = posurines_df.iloc[relevant_samples]
            posurines_classes = [int(np.argwhere(l == self.unique_labels).flatten()) for l in posurines_df.loc[:, 'labels'].to_numpy()]
            posurines_preds = [int(np.argwhere(l == self.unique_labels).flatten()) for l in posurines_df.loc[:, 'preds'].to_numpy()]

            lists['acc']['posurines'] = [ACC(posurines_preds, posurines_classes)]
            lists['mcc']['posurines'] = [MCC(posurines_preds, posurines_classes)]
            lists['classes']['posurines'] = [posurines_classes for _ in range(len(lists['preds']['posurines']))]
            lists['preds']['posurines'] = [posurines_preds for _ in range(len(lists['preds']['posurines']))]
            lists['names']['posurines'] = [posurines_df.loc[:, 'names'].to_numpy() for _ in range(len(lists['preds']['posurines']))]
            lists['batches']['posurines'] = [posurines_df.loc[:, 'batches'].to_numpy() for _ in range(len(lists['preds']['posurines']))]
            lists['labels']['posurines'] = [posurines_df.loc[:, 'labels'].to_numpy() for _ in range(len(lists['preds']['posurines']))]
            lists['proba']['posurines'] = [posurines_df.loc[:, 'proba'].to_numpy() for _ in range(len(lists['preds']['posurines']))]

            fig = get_confusion_matrix(posurines_classes, posurines_preds, self.unique_labels)
            save_confusion_matrix(fig, 
                                    f"{self.log_path}/confusion_matrices/" 
                                    f"{self.name}_{self.args.model_name}_posurines", 
                                    acc=lists['acc']['posurines'], mcc=lists['mcc']['posurines'], group='posurines')
            run[f'confusion_matrix/posurines'].upload(fig)

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



