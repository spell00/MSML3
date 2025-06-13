import os
# NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
# NEPTUNE_PROJECT_NAME = "Bacteria-MSMS"
# NEPTUNE_MODEL_NAME = 'BAC-'
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZjRiZGUzYS1kNTJmLTRkNGItOWU1MS1iNDU3MGE1NjAyODAifQ=="
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'
import pandas as pd
import numpy as np
import random
import json
import sklearn.neighbors
import torch
import sklearn
import os
import pickle
import neptune
from scipy import stats
import multiprocessing

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from utils import scale_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
# import ACC and MCC
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import matthews_corrcoef as MCC

warnings.filterwarnings("ignore")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# from rpy2.robjects.packages import importr
# from rpy2.robjects.conversion import localconverter
import copy
from skopt import gp_minimize
from sklearn_train_nocv import count_labels, get_confusion_matrix, save_roc_curve, plot_roc
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.multiclass import OneVsRestClassifier
# from sklearn import metrics
from loggings import log_ord, log_fct, save_confusion_matrix, log_neptune
from utils import augment_data, get_empty_lists, softmax, remove_zero_cols
from dataset import get_data
# from features_selection_sparse import keep_not_zeros_sparse, MultiKeepNotFunctionsSparse, split_sparse


class Train:
    def __init__(self, name, model, data, uniques, hparams_names, log_path, args, logger, log_neptune, mlops='None'):
        self.log_neptune = log_neptune
        self.best_roc_score = -1
        self.args = args
        self.log_path = log_path
        self.model = model
        self.data = data
        self.logger = logger
        self.hparams_names = hparams_names
        self.best_scores = {
            'acc': {
                'train': -1,
                'valid': -1,
                'test': -1,
                'posurines': -1,
            },
            'mcc': {
                'train': -1,
                'valid': -1,
                'test': -1,
                'posurines': -1,
            }
        }
        self.iter = 0
        self.model = model
        self.name = name
        self.mlops = mlops
        self.uniques = uniques
        self.best_params_dict = {}
        self.best_params_dict_values = {}
        if self.args.binary:
            self.data['labels']['all'] = np.array(['blanc' if label=='blanc' else 'bact' for label in self.data['labels']['all']])
        self.unique_labels = np.array(np.unique(self.data['labels']['all']))
        # place blancs at the end
        blanc_class = np.argwhere(self.unique_labels == 'blanc').flatten()[0]
        self.unique_labels = np.concatenate((['blanc'], np.delete(self.unique_labels, blanc_class)))
        self.uniques2 = copy.deepcopy(self.uniques)
        self.uniques2['labels'] = None



    def train(self, h_params):
        metrics = {}
        self.iter += 1
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

        # other_params = {'p': p, 'g': g, 'n_aug': n_aug}

        lists = get_empty_lists()

        all_data = {
            'inputs': copy.deepcopy(self.data['inputs']),
            'labels': copy.deepcopy(self.data['labels']),
            'batches': copy.deepcopy(self.data['batches']),
            'concs': copy.deepcopy(self.data['concs']),
            'names': copy.deepcopy(self.data['names']),
        }

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
            model["csv_file"] = run["csv_file"] = args.csv_file
            model["model_name"] = run["model_name"] = f'binary{self.args.binary}_{self.args.model_name}'
            model["groupkfold"] = run["groupkfold"] = args.groupkfold
            model["dataset_name"] = run["dataset_name"] = 'MSML-Bacteria'
            model["scaler_name"] = run["scaler_name"] = scaler_name
            model["mz_min"] = run["mz_min"] = args.min_mz
            model["mz_max"] = run["mz_max"] = args.max_mz
            model["rt_min"] = run["rt_min"] = args.min_rt
            model["rt_max"] = run["rt_max"] = args.max_rt
            model["mz_bin"] = run["mz_bin"] = args.mz
            model["rt_bin"] = run["rt_bin"] = args.rt
            model["path"] = run["path"] = self.log_path
            model["concs"] = run["concs"] = args.concs
            model["binary"] = run["binary"] = args.binary
            model['spd'] = run['spd'] = args.spd
            model['ovr'] = run['ovr'] = args.ovr
            model['train_on'] = run['train_on'] = args.train_on
            model['n_features'] = run['n_features'] = args.n_features
            model['total_features'] = run['total_features'] = all_data['inputs']['all'].shape[1]
            model['ms_level'] = run['ms_level'] = args.ms_level
            model['log'] = run['log'] = args.log

        else:
            model = None
            run = None

        all_data, scaler = scale_data(scaler_name, all_data)

        # save scaler
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/{scaler_name}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        print(f'Iteration: {self.iter}')
        # combinations = []
        models = []
        h = 0
        seed = 0

        if self.args.groupkfold:
            self.args.n_repeats = len(np.unique(all_data['batches']['all']))
        while h < self.args.n_repeats:
            lists['names']['posurines'] += [np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])]
            lists['batches']['posurines'] += [all_data['batches']['urinespositives']]
            if self.args.train_on == 'all':
                if self.args.groupkfold:
                    # skf = GroupShuffleSplit(n_splits=5, random_state=seed)
                    skf = StratifiedGroupKFold(n_splits=len(np.unique(all_data['batches']['all'])), shuffle=True, random_state=seed)
                    train_nums = np.arange(0, len(all_data['labels']['all']))
                    splitter = skf.split(train_nums, self.data['labels']['all'], all_data['batches']['all'])
                else:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=h)
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

                train_data, valid_data, test_data = all_data['inputs']['all'].iloc[train_inds], all_data['inputs']['all'].iloc[valid_inds], all_data['inputs']['all'].iloc[test_inds]
                train_labels, valid_labels, test_labels = all_data['labels']['all'][train_inds], all_data['labels']['all'][valid_inds], all_data['labels']['all'][test_inds]
                train_batches, valid_batches, test_batches = all_data['batches']['all'][train_inds], all_data['batches']['all'][valid_inds], all_data['batches']['all'][test_inds]

                # remove labels that are not in train
                unique_train_labels = np.unique(train_labels)
                valid_to_keep = np.array(
                    [i for i, l in enumerate(valid_labels) if l in unique_train_labels ]
                )
                test_to_keep = np.array(
                    [i for i, l in enumerate(test_labels) if l in unique_train_labels]
                )
                valid_data, valid_labels, valid_batches = valid_data.iloc[valid_to_keep], valid_labels[valid_to_keep], valid_batches[valid_to_keep]
                test_data, test_labels, test_batches = test_data.iloc[test_to_keep], test_labels[test_to_keep], test_batches[test_to_keep]
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

                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
                valid_nums = np.arange(0, len(valid_labels))
                splitter = skf.split(valid_nums, valid_labels, valid_batches)
                test_inds, valid_inds = splitter.__next__()
                test_data, valid_data = valid_data.iloc[test_inds], valid_data.iloc[valid_inds]
                test_labels, valid_labels = valid_labels[test_inds], valid_labels[valid_inds]
                test_batches, valid_batches = valid_batches[test_inds], valid_batches[valid_inds]
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

                skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=h)
                valid_nums = np.arange(0, len(valid_labels))
                splitter = skf.split(valid_nums, valid_batches, valid_batches)
                test_inds, valid_inds = splitter.__next__()
                test_data, valid_data = valid_data.iloc[test_inds], valid_data.iloc[valid_inds]
                test_labels, valid_labels = valid_labels[test_inds], valid_labels[valid_inds]
                test_batches, valid_batches = valid_batches[test_inds], valid_batches[valid_inds]

            lists['inds']['train'] += [train_inds]
            lists['inds']['valid'] += [valid_inds]
            lists['inds']['test'] += [test_inds]

            lists['batches']['train'] += [train_batches]
            lists['batches']['valid'] += [valid_batches]
            lists['batches']['test'] += [test_batches]
            lists['unique_batches']['train'] += [list(np.unique(train_batches))]
            lists['unique_batches']['valid'] += [list(np.unique(valid_batches))]
            lists['unique_batches']['test'] += [list(np.unique(test_batches))]

            if n_aug > 0:
                train_data = augment_data(train_data, n_aug, p, g)
                train_data = np.nan_to_num(train_data)
                train_labels = np.concatenate([train_labels] * (n_aug + 1))
                train_batches = np.concatenate([train_batches] * (n_aug + 1))
            else:
                train_data = train_data.fillna(0)
            valid_data = valid_data.fillna(0)
            test_data = test_data.fillna(0)

            lists['classes']['train'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in train_labels])]
            lists['classes']['valid'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in valid_labels])]
            lists['classes']['test'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in test_labels])]
            lists['labels']['train'] += [train_labels]
            lists['labels']['valid'] += [valid_labels]
            lists['labels']['test'] += [test_labels]

            m = self.model()
            m.set_params(**param_grid)
            if self.args.ovr:
                m = OneVsRestClassifier(m)
            m.fit(train_data, lists['classes']['train'][-1])
            models += [m]
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
            except:
                pass
            lists['mcc']['train'] += [MCC(lists['classes']['train'][-1], lists['preds']['train'][-1])]
            lists['mcc']['valid'] += [MCC(lists['classes']['valid'][-1], lists['preds']['valid'][-1])]
            lists['mcc']['test'] += [MCC(lists['classes']['test'][-1], lists['preds']['test'][-1])]
                
            if self.best_scores['acc']['valid'] is None:
                self.best_scores['acc']['valid'] = 0
            h += 1

        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        log_ord(self.data, uniques, ord_path, scaler_name, run)
        data = copy.deepcopy(self.data)
        metrics = log_fct(data, scaler_name, metrics)
        log_ord(data, self.uniques2, ord_path, f'{scaler_name}_blancs', run)

        print(lists['mcc']['valid'])
        print('valid_score:', np.mean(lists['acc']['valid']), \
              'valid_mcc:', np.mean(lists['mcc']['valid']), \
              'scaler:', scaler_name,
              'h_params:', param_grid
              )
        lists = self.save_confusion_matrices(all_data, lists, run)
        self.save_roc_curves(lists, run)
        if np.mean(lists['mcc']['valid']) > np.mean(self.best_scores['mcc']['valid']):
            self.dump_models(models, lists)
            # Save the individual scores of each sample with class, #batch
            self.save_results_df(lists, run)
            self.retrieve_best_scores(lists)

        if self.log_neptune:
            log_neptune(run, lists)
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

        with open(f'{self.log_path}/saved_models/unique_batches_{self.name}_{self.args.model_name}.json', "w") as read_file:
            json.dump(unique_batches, read_file)

        self.best_params_dict_values['train_batches'] = self.best_scores_train

        self.best_params_dict_values['train_acc'] = self.best_scores_train
        self.best_params_dict_values['valid_acc'] = self.best_scores['acc']['valid']
        self.best_params_dict_values['test_acc'] = self.best_scores['acc']['test']
        self.best_params_dict_values['train_acc'] = self.best_scores_train
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
        n_batches = np.unique(np.concatenate([np.concatenate(unique_batches[b]) for b in unique_batches])).flatten().shape[0]
        self.best_params_dict['nBE'] = (np.log(n_batches) - metrics[scaler_name]['all']['shannon']['domains'])  / np.log(n_batches)

        self.best_params_dict['train_acc_mean'] = np.mean(self.best_scores_train)
        self.best_params_dict['valid_acc_mean'] = np.mean(self.best_scores['acc']['valid'])
        self.best_params_dict['test_acc_mean'] = np.mean(self.best_scores['acc']['test'])
        self.best_params_dict['train_acc_std'] = np.std(self.best_scores_train)
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
        # load model

    def make_predictions(self, all_data, lists, run):
        urinespositives_names = np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])
        urinespositives_batches = np.array([x for x in all_data['batches']['urinespositives']])
        # TODO load expected classes elsewhere
        urinespositives_real_df = pd.read_csv(f'resources/bacteries_2024/B10-05-03-2024/b10_patients_samples.csv')
        urinespositives_real_df.loc[:, 'Class'] = [l.lower() for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]
        if self.args.binary:
            urinespositives_real_df.loc[:, 'Class'] = ['blanc' if l == 'blanc' else 'bact' for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]

        to_keep = np.argwhere(np.isin(urinespositives_names, urinespositives_real_df.loc[:, 'ID'].to_numpy()) == True).flatten()
        urinespositives_names = urinespositives_names[to_keep]
        urinespositives_batches = urinespositives_batches[to_keep]
        lists['preds']['posurines'] = [[lists['preds']['posurines'][j][i] for i in to_keep] for j in range(len(lists['preds']['posurines']))]
        lists['proba']['posurines'] = [[lists['proba']['posurines'][j][i] for i in to_keep] for j in range(len(lists['proba']['posurines']))]
        # make the order of  urinespositives_real_df.loc[:, 'Class'] the same as urinespositives_names
        new_order = np.argsort(urinespositives_names)
        urinespositives_real_df = urinespositives_real_df.iloc[new_order]
        # make sure the order is the same
        assert np.sum([x == urinespositives_real_df.loc[:, 'ID'].to_numpy() for x in urinespositives_names]) == len(urinespositives_names)
        try:
            posurines_df = pd.DataFrame(
                {
                    'names': np.concatenate([urinespositives_names for _ in range(len(lists['preds']['posurines']))]),
                    'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(lists['preds']['posurines'])]),
                    'batches': np.concatenate([urinespositives_batches for _ in range(len(lists['preds']['posurines']))]),
                    'preds': np.concatenate(lists['preds']['posurines']),
                    'labels': np.concatenate([urinespositives_real_df.loc[:, 'Class'] for _ in range(len(lists['preds']['posurines']))]),                
                    'proba': np.concatenate(lists['proba']['posurines']).max(1),                    
                }
            )
        except:
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
        assert 'bact' not in posurines_df.loc[:, 'preds']
        posurines_df.to_csv(
            f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
        )
        # save the mode scores too
        preds_posurines = np.stack(lists['preds']['posurines'])
        preds_posurines = stats.mode(preds_posurines, axis=0)[0].flatten()

        try:
            proba_posurines = np.mean(np.stack(lists['proba']['posurines']), 0)
            proba_posurines = np.array([
                proba_posurines[i, x] for i, x in enumerate(preds_posurines)
            ])
            posurines_df = pd.DataFrame(
                {
                    'names': urinespositives_names,
                    'batches': urinespositives_batches,
                    'preds': preds_posurines,
                    'labels': urinespositives_real_df.loc[:, 'Class'],                
                    'proba': proba_posurines,                    
                }
            )
        except:
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

        return posurines_df

    def save_results_df(self, lists, run):
        if len(lists['proba']['train']) > 0:
            df_valid = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['valid']),
                    'labels': np.concatenate(lists['labels']['valid']),
                    'batches': np.concatenate(lists['batches']['valid']),
                    'preds': np.concatenate(lists['preds']['valid']), 
                    'proba': np.concatenate(lists['proba']['valid']).max(1)
                }
            )
            df_test = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['test']),
                    'labels': np.concatenate(lists['labels']['test']),
                    'batches': np.concatenate(lists['batches']['test']),
                    'preds': np.concatenate(lists['preds']['test']),
                    'proba': np.concatenate(lists['proba']['test']).max(1)
                }
            )

        else:
            df_valid = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['valid']),
                    'labels': np.concatenate(lists['labels']['valid']),
                    'batches': np.concatenate(lists['batches']['valid']),
                    'preds': np.concatenate(lists['preds']['valid']), 
                }
            )
            df_test = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['test']),
                    'labels': np.concatenate(lists['labels']['test']),
                    'batches': np.concatenate(lists['batches']['test']),
                    'preds': np.concatenate(lists['preds']['test']),
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

    def dump_models(self, models, lists):
        for i, m in enumerate(models):
            # save model
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}.pkl', 'wb') as f:
                pickle.dump(m, f)
            # save indices
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}_train_indices.pkl', 'wb') as f:
                pickle.dump(lists['inds']['train'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}_valid_indices.pkl', 'wb') as f:
                pickle.dump(lists['inds']['valid'][i], f)
            with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}_test_indices.pkl', 'wb') as f:
                pickle.dump(lists['inds']['test'][i], f)

    def retrieve_best_scores(self, lists):
        self.best_scores['acc']['train'] = lists['acc']['train']
        self.best_scores['acc']['valid'] = lists['acc']['valid']
        self.best_scores['acc']['test'] = lists['acc']['test']
        self.best_scores['mcc']['train'] = lists['mcc']['train']
        self.best_scores['mcc']['valid'] = lists['mcc']['valid']
        self.best_scores['mcc']['test'] = lists['mcc']['test']
        
    def save_confusion_matrices(self, all_data, lists, run):
        posurines_df = self.make_predictions(all_data, lists, run)

        posurines_classes = [int(np.argwhere(l == self.unique_labels).flatten()) for l in posurines_df.loc[:, 'labels'].to_numpy()]
        posurines_preds = [int(np.argwhere(l == self.unique_labels).flatten()) for l in posurines_df.loc[:, 'preds'].to_numpy()]

        lists[f'acc']['posurines'] = [ACC(posurines_preds, posurines_classes)]
        lists[f'mcc']['posurines'] = [MCC(posurines_preds, posurines_classes)]
        lists['classes']['posurines'] = [posurines_classes]
        lists['preds']['posurines'] = [posurines_preds]

        for group in ['train', 'valid', 'test', 'posurines']:
            fig = get_confusion_matrix(np.concatenate(lists['classes'][group]), 
                                        np.concatenate(lists['preds'][group]), 
                                        self.unique_labels)
            save_confusion_matrix(fig, 
                                    f"{self.log_path}/confusion_matrices/" 
                                    f"{self.name}_{self.args.model_name}_{group}", 
                                    acc=lists['acc'][group], mcc=lists['mcc'][group], group=group)
            run[f'confusion_matrix/{group}'].upload(fig)


        return lists



if __name__ == '__main__':
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, default='mutual_info_classif_scores.csv')
    parser.add_argument('--remove_zeros', type=int, default=0)
    parser.add_argument('--log1p', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--pool', type=int, default=0)
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='linsvc')
    parser.add_argument('--train_on', type=str, default='all')
    parser.add_argument('--csv_file', type=str, default='inputs.csv')
    parser.add_argument('--ovr', type=int, default=1)
    parser.add_argument('--mz', type=int, default=10)
    parser.add_argument('--rt', type=int, default=10)
    parser.add_argument('--mzp', type=int, default=10)
    parser.add_argument('--rtp', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--spd', type=int, default=200)
    parser.add_argument('--ms_level', type=int, default=2)
    parser.add_argument('--combat', type=int, default=0)  # TODO not using this anymore
    parser.add_argument('--shift', type=int, default=0)  # TODO keep this?
    parser.add_argument('--log', type=str, default='inloop')
    parser.add_argument('--features_selection', type=str, default='mutual_info_classif')
    parser.add_argument('--concs', type=str, default='na,h')
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument("--min_mz", type=int, default=0)
    parser.add_argument("--max_mz", type=int, default=10000)
    parser.add_argument("--min_rt", type=int, default=0)
    parser.add_argument("--max_rt", type=int, default=1000)
    args = parser.parse_args()

    batch_dates = [
        'B10-05-03-2024',
        'B1-02-02-2024', 'B2-02-21-2024', 'B3-02-29-2024', 
        'B4-03-01-2024', 'B5-03-13-2024', 'B6-03-29-2024',
        'B7-04-03-2024', "B8-04-15-2024", "B9-04-22-2024",
    ]
    batches_to_keep = [
        'b10-05-03-2024',
        'b1-02-02-2024', 'b2-02-21-2024', 'b3-02-29-2024',
        'b4-03-01-2024', 'b5-03-13-2024', 'b6-03-29-2024',
        'b7-04-03-2024', 'b8-04-15-2024', 'b9-04-22-2024',
    ]
    batch_dates = [x.split('-')[0] for x in batch_dates]
    batches_to_keep = [x.split('-')[0] for x in batches_to_keep]

    concs = args.concs.split(',')
    cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    exp_name = f'{"-".join(batches_to_keep)}_binary{args.binary}_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_{cropings}_{"_".join(concs)}'

    # TODO change gkf0; only valid because using all features
    exp = f'all_{"-".join(batch_dates)}_gkf0_{cropings}_5splits'  
    path = f'resources/bacteries_2024/matrices/mz{args.mz}/rt{args.rt}/' \
        f'mzp{args.mzp}/rtp{args.rtp}/thr{args.threshold}/{args.spd}spd/' \
        f'ms{args.ms_level}/combat{args.combat}/shift{args.shift}/none/' \
        f'log{args.log}/{args.features_selection}/{exp}'

    data, unique_labels, unique_batches, unique_manips, \
                        unique_urines, unique_concs = get_data(path, args)

    uniques = {
        'labels': unique_labels,
        'batches': unique_batches,
        'manips': unique_manips,
        'urines': unique_urines,
        'concs': unique_concs
    }
    # uniques2 = {
    #     'labels': None,
    #     'batches': unique_batches,
    #     'manips': unique_manips,
    #     'urines': unique_urines,
    #     'concs': unique_concs
    # }

    mask = [True if x in batches_to_keep else False for x in data['batches']['all']]
    data['inputs']['all'] = data['inputs']['all'].iloc[mask]
    data['labels']['all'] = data['labels']['all'][mask]
    data['batches']['all'] = data['batches']['all'][mask]
    data['names']['all'] = data['names']['all'][mask]
    data['orders']['all'] = data['orders']['all'][mask]
    data['cats']['all'] = data['cats']['all'][mask]
    data['urines']['all'] = data['urines']['all'][mask]
    data['manips']['all'] = data['manips']['all'][mask]
    data['concs']['all'] = data['concs']['all'][mask]

    mask2 = [True if x in concs else False for x in data['concs']['all']]
    data['inputs']['all'] = data['inputs']['all'].iloc[mask2]
    data['labels']['all'] = data['labels']['all'][mask2]
    data['batches']['all'] = data['batches']['all'][mask2]
    data['names']['all'] = data['names']['all'][mask2]
    data['orders']['all'] = data['orders']['all'][mask2]
    data['cats']['all'] = data['cats']['all'][mask2]
    data['urines']['all'] = data['urines']['all'][mask2]
    data['manips']['all'] = data['manips']['all'][mask2]
    data['concs']['all'] = data['concs']['all'][mask2]
    
    path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/{args.spd}spd/thr{args.threshold}/' \
           f'{args.train_on}/{exp_name}/{args.model_name}/'

    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()
    standard_minmax_scaler = Pipeline([('standard', StandardScaler()),
                                       ('minmax', MinMaxScaler())])
    robust_minmax_scaler = Pipeline([('robust', RobustScaler()),
                                     ('minmax', MinMaxScaler())])

    if args.model_name == 'RF':
        cfr = RandomForestClassifier
        space = [
            # Integer(100, 20000, 'uniform', name='features_cutoff'),
            Real(0, 1, 'uniform', name='threshold'),
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(1, 100, 'uniform', name="max_features"),
            Integer(2, 10, 'uniform', name="min_samples_split"),
            Integer(1, 10, 'uniform', name="min_samples_leaf"),
            Integer(100, 1000, 'uniform', name="n_estimators"),
            Categorical(['gini', 'entropy'], name="criterion"),
            Categorical([True, False], name="oob_score"),
            Categorical(['balanced'], name="class_weight"),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler"),
        ]
    elif args.model_name == 'NB':
        from sklearn.naive_bayes import GaussianNB
        cfr = GaussianNB
        space = [
            # Integer(100, 20000, 'uniform', name='features_cutoff'),
            # Real(0, 1, 'uniform', name='threshold'),
            Integer(0, 5, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler"),
        ]
    elif args.model_name == 'linsvc':
        cfr = sklearn.svm.LinearSVC
        space = [
            Real(0, 1, 'uniform', name='threshold'),
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Real(1e-4, 10, 'log-uniform', name='tol'),
            Integer(3000, 10000, 'uniform', name='max_iter'),
            Categorical(['l2'], name='penalty'),
            Categorical(['hinge', 'squared_hinge'], name='loss'),
            Real(1e-5, 1, 'uniform', name='C'),
            Categorical(['balanced'], name='class_weight'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler")
        ]
    elif args.model_name == 'knn':
        cfr = sklearn.neighbors.KNeighborsClassifier
        space = [
            Real(0, 1, 'uniform', name='threshold'),
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(0, 10, name='n_neighbors'),
            Categorical(['uniform', 'balanced'], name='weights'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler")
        ]
    elif args.model_name == 'svclinear':
        cfr = sklearn.svm.SVC
        space = [
            Real(0, 1, 'uniform', name='threshold'),
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Real(1e-4, 10, 'log-uniform', name='tol'),
            Integer(1, 10000, 'uniform', name='max_iter'),
            Real(1e-3, 10000, 'uniform', name='C'),
            Categorical(['balanced'], name='class_weight'),
            Categorical(['linear'], name='kernel'),
            Categorical([True], name='probability'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler")
        ]
    elif args.model_name == 'LDA':
        cfr = LDA
        space = [
            Real(0, 1, 'uniform', name='threshold'),
            Integer(0, 5, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler"),
        ]
    elif args.model_name == 'logreg':
        cfr = sklearn.linear_model.LogisticRegression
        space = [
            Real(0, 1, 'uniform', name='threshold'),
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(1, 100, 'uniform', name='max_iter'),
            Real(1e-3, 20000, 'uniform', name='C'),
            Categorical(['saga'], name='solver'),
            Categorical(['l1', 'l2'], name='penalty'),
            Categorical([True, False], name='fit_intercept'),
            Categorical(['balanced'], name='class_weight'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler"),
        ]

    hparams_names = [x.name for x in space]
    train = Train(name="inputs", model=cfr, data=data, uniques=uniques,
                  hparams_names=hparams_names, log_path=path, args=args, 
                  logger=None, log_neptune=True, mlops='None')
    
    res = gp_minimize(train.train, space, n_calls=30, random_state=1)

    from skopt.plots import plot_objective, plot_histogram, plot_convergence, plot_evaluations, plot_regret
    os.makedirs(f'{path}/figures/skopt/', exist_ok=True)
    plot_convergence(res)
    plt.savefig(f'{path}/figures/skopt/convergence.png')
    try:
        plot_evaluations(res)
        plt.savefig(f'{path}/figures/skopt/evaluations.png')
    except:
        pass
    try:
        plot_objective(res)
        plt.savefig(f'{path}/figures/skopt/objective.png')
    except:
        pass
    try:
        plot_regret(res)
        plt.savefig(f'{path}/figures/skopt/regret.png')
    except:
        pass
