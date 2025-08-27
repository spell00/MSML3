#!/usr/bin/python3
import os
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZjRiZGUzYS1kNTJmLTRkNGItOWU1MS1iNDU3MGE1NjAyODAifQ=="
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-2'

import pandas as pd
import numpy as np
import random
import json
import sklearn.neighbors
import torch
import sklearn
import pickle
import neptune

from utils import remove_zero_cols, scale_data, augment_data, get_empty_lists

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

import copy

from skopt import gp_minimize
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
from sklearn.multiclass import OneVsRestClassifier
# from sklearn import metrics
from loggings import log_ord, log_fct, log_neptune
from dataset import get_data
# from sklearn_train_nocv import count_labels, get_confusion_matrix, save_roc_curve, plot_roc
from train import Train
from log_shap import log_shap

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Train2Models(Train):
    def __init__(self, name, model, data, uniques, hparams_names, log_path, 
                 args, logger, log_neptune, mlops='None', binary_path=None):
        super().__init__(name, model, data, uniques, hparams_names, log_path, 
                         args, logger, log_neptune, mlops, binary_path)
        self.model_name = f'2models_{self.args.model_name}'

    def make_combined_proba(self, lists):
        for group in lists['binary_proba']:
            if len(lists['binary_proba'][group]) == 0:
                continue
            tmp = np.zeros((len(lists['binary_proba'][group][-1]), len(self.unique_labels)))
            for i, arr in enumerate(lists['binary_proba'][group][-1]):
                tmp[i][0] = arr[0]
                tmp[i][-1] = arr[1]
            lists['binary_proba'][group][-1] = tmp
        
        for group in lists['proba']:
            if len(lists['proba'][group]) == 0:
                continue
            tmp = np.zeros((len(lists['proba'][group][-1]), len(self.unique_labels)))
            for i, arr in enumerate(lists['proba'][group][-1]):
                tmp[i][1:-1] = arr
            lists['proba'][group][-1] = tmp
        
        return lists

    def train(self, h_params):
        # NORMALIZE BINARY DATA
        # import binary model scaler
        # TODO this should be in init
        json_file = f'{self.binary_path}/saved_models/best_params_inputs_{self.args.binary_model_name}.json'
        with open(json_file, 'r') as f:
            binary_params = json.load(f)
            bscaler_name = binary_params['scaler']
        with open(f'{self.binary_path}/saved_models/{bscaler_name}_scaler.pkl', 'rb') as f:
            bscaler = pickle.load(f)

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
                model=f'{NEPTUNE_MODEL_NAME}{self.args.model_name.upper()}',
                project=NEPTUNE_PROJECT_NAME,
                api_token=NEPTUNE_API_TOKEN,
            )
            model['hparams'] = run["hparams"] = hparams
            model["csv_file"] = run["csv_file"] = args.csv_file
            model["model_name"] = run["model_name"] = self.model_name
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
            model['ms_level'] = run['ms_level'] = args.ms_level
            model['log'] = run['log'] = args.log
            model["remove_blancs"] = run["remove_blancs"] = args.remove_blancs
            model["binary_model_name"] = run["binary_model_name"] = args.binary_model_name
            model["binary_path"] = run["binary_path"] = self.binary_path
            model['batches'] = run['batches'] = '-'.join(self.uniques['batches'])
        else:
            model = None
            run = None


        other_params = {
            'features_cutoff': features_cutoff,
            'threshold': threshold,
            'n_aug': n_aug,
            'p': p,
            'g': g,
        }

        lists = get_empty_lists()

        all_data_binary = {
            'inputs': copy.deepcopy(self.data['inputs']),
            'labels': copy.deepcopy(self.data['labels']),
            'batches': copy.deepcopy(self.data['batches']),
            'concs': copy.deepcopy(self.data['concs']),
            'names': copy.deepcopy(self.data['names']),
        }
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

        not_zeros_col = remove_zero_cols(all_data_binary['inputs']['all'], binary_params['threshold'])
        all_data_binary['inputs']['all'] = all_data_binary['inputs']['all'].iloc[:, not_zeros_col]
        all_data_binary['inputs']['train'] = all_data_binary['inputs']['train'].iloc[:, not_zeros_col]
        all_data_binary['inputs']['valid'] = all_data_binary['inputs']['valid'].iloc[:, not_zeros_col]
        all_data_binary['inputs']['test'] = all_data_binary['inputs']['test'].iloc[:, not_zeros_col]
        all_data_binary['inputs']['urinespositives'] = all_data_binary['inputs']['urinespositives'].iloc[:, not_zeros_col]

        if self.args.log_neptune:
            model['total_features'] = run['total_features'] = self.data['inputs']['all'].shape[1]


        self.unique_labels = np.array(np.unique(all_data['labels']['all']))
        # place blancs at the end
        blanc_class = np.argwhere(self.unique_labels == 'blanc').flatten()[0]
        self.unique_labels = np.concatenate((['blanc'], np.delete(self.unique_labels, blanc_class)))
       
        all_binary_labels = pd.Series(['blanc' if label == 'blanc' else 'bact' for label in all_data['labels']['all']])	
        unique_binary_labels = np.array(np.unique(all_binary_labels))
        unique_binary_labels = np.concatenate((['blanc'], np.delete(unique_binary_labels, np.argwhere(unique_binary_labels == 'blanc').flatten()[0])))

        # NORMALIZE DATA
        all_data, scaler = scale_data(scaler_name, all_data)
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/{scaler_name}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        if bscaler_name in ['minmax2', 'l2', 'l1', 'zscore']:
            all_data_binary, scaler = scale_data(bscaler_name, all_data_binary)
        else:
            all_data_binary['inputs']['all'].iloc[:] = bscaler.transform(all_data_binary['inputs']['all'])
            all_data_binary['inputs']['train'].iloc[:] = bscaler.transform(all_data_binary['inputs']['train'])
            all_data_binary['inputs']['valid'].iloc[:] = bscaler.transform(all_data_binary['inputs']['valid'])
            all_data_binary['inputs']['test'].iloc[:] = bscaler.transform(all_data_binary['inputs']['test'])
            all_data_binary['inputs']['urinespositives'].iloc[:] = bscaler.transform(all_data_binary['inputs']['urinespositives'])
        
        print(f'Iteration: {self.iter}')
        combinations = []
        models = []
        h = 0
        seed = 0
        if self.args.groupkfold:
            self.args.n_repeats = len(np.unique(all_data['batches']['all']))
        while h < self.args.n_repeats:
            lists['names']['posurines'] += [np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])]
            lists['batches']['posurines'] += [all_data['batches']['urinespositives']]
            # import train valid and test indices
            with open(f'{self.binary_path}/saved_models/{self.args.binary_model_name}_{h}_train_indices.pkl', 'rb') as f:
                train_inds = pickle.load(f)
            with open(f'{self.binary_path}/saved_models/{self.args.binary_model_name}_{h}_valid_indices.pkl', 'rb') as f:
                valid_inds = pickle.load(f)
            with open(f'{self.binary_path}/saved_models/{self.args.binary_model_name}_{h}_test_indices.pkl', 'rb') as f:
                test_inds = pickle.load(f)
            # import binary best model
            with open(f'{self.binary_path}/saved_models/{self.args.binary_model_name}_{h}.pkl', 'rb') as f:
                binary_model = pickle.load(f)
            
            train_data, valid_data, test_data = all_data['inputs']['all'].iloc[train_inds], \
                all_data['inputs']['all'].iloc[valid_inds], all_data['inputs']['all'].iloc[test_inds]
            train_labels, valid_labels, test_labels = all_data['labels']['all'][train_inds], \
                all_data['labels']['all'][valid_inds], all_data['labels']['all'][test_inds]
            train_batches, valid_batches, test_batches = all_data['batches']['all'][train_inds], \
                all_data['batches']['all'][valid_inds], all_data['batches']['all'][test_inds]
            train_binary_data, valid_binary_data, test_binary_data = all_data_binary['inputs']['all'].iloc[train_inds], \
                  all_data_binary['inputs']['all'].iloc[valid_inds], all_data_binary['inputs']['all'].iloc[test_inds]
            train_binary_data = train_binary_data.fillna(0)
            valid_binary_data = valid_binary_data.fillna(0)
            test_binary_data = test_binary_data.fillna(0)

            lists['inds']['train'] += [train_inds]
            lists['inds']['valid'] += [valid_inds]
            lists['inds']['test'] += [test_inds]

            lists['batches']['train'] += [train_batches]
            lists['batches']['valid'] += [valid_batches]
            lists['batches']['test'] += [test_batches]

            lists['unique_batches']['train'] += [list(np.unique(train_batches))]
            lists['unique_batches']['valid'] += [list(np.unique(valid_batches))]
            lists['unique_batches']['test'] += [list(np.unique(test_batches))]

            train_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in train_labels])
            valid_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in valid_labels])
            test_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in test_labels])
            lists['binary_labels']['train'] += [train_binary_labels]
            lists['binary_labels']['valid'] += [valid_binary_labels]
            lists['binary_labels']['test'] += [test_binary_labels]

            lists['binary_classes']['train'] += [
                np.array(
                    [np.argwhere(l == unique_binary_labels).flatten() for l in train_binary_labels]
                )
            ]
            lists['binary_classes']['valid'] += [
                np.array(
                    [np.argwhere(l == unique_binary_labels).flatten() for l in valid_binary_labels]
                )
            ]
            lists['binary_classes']['test'] += [
                np.array(
                    [np.argwhere(l == unique_binary_labels).flatten() for l in test_binary_labels]
                    )
            ]

            lists['binary_preds']['train'] += [binary_model.predict(train_binary_data)]
            lists['binary_preds']['valid'] += [binary_model.predict(valid_binary_data)]
            lists['binary_preds']['test'] += [binary_model.predict(test_binary_data)]
            try:
                lists['binary_proba']['train'] += [binary_model.predict_proba(train_binary_data)]
                lists['binary_proba']['valid'] += [binary_model.predict_proba(valid_binary_data)]
                lists['binary_proba']['test'] += [binary_model.predict_proba(test_binary_data)]
            except:
                pass

            if self.args.remove_blancs:
                blanc_class = np.argwhere(self.unique_labels == 'blanc').flatten()[0]
                # drop blanc class in unique_labels
                blanc_binary_class = np.argwhere(unique_binary_labels == 'blanc').flatten()[0]
                blanc_bact_class = np.argwhere(unique_binary_labels != 'blanc').flatten()[0]
                train_not_blanc_preds = np.argwhere(lists['binary_preds']['train'][-1] != blanc_binary_class).flatten()
                valid_not_blanc_preds = np.argwhere(lists['binary_preds']['valid'][-1] != blanc_binary_class).flatten()
                test_not_blanc_preds = np.argwhere(lists['binary_preds']['test'][-1] != blanc_binary_class).flatten()

                # keep only predictions that are blancs
                lists['binary_classes']['train'][-1] = np.delete(lists['binary_classes']['train'][-1], train_not_blanc_preds)
                lists['binary_classes']['valid'][-1] = np.delete(lists['binary_classes']['valid'][-1], valid_not_blanc_preds)
                lists['binary_classes']['test'][-1] = np.delete(lists['binary_classes']['test'][-1], test_not_blanc_preds)
                lists['binary_preds']['train'][-1] = np.delete(lists['binary_preds']['train'][-1], train_not_blanc_preds)
                lists['binary_preds']['valid'][-1] = np.delete(lists['binary_preds']['valid'][-1], valid_not_blanc_preds)
                lists['binary_preds']['test'][-1] = np.delete(lists['binary_preds']['test'][-1], test_not_blanc_preds)
                lists['binary_labels']['train'][-1] = np.delete(lists['binary_labels']['train'][-1].to_numpy(), train_not_blanc_preds)
                lists['binary_labels']['valid'][-1] = np.delete(lists['binary_labels']['valid'][-1].to_numpy(), valid_not_blanc_preds)
                lists['binary_labels']['test'][-1] = np.delete(lists['binary_labels']['test'][-1].to_numpy(), test_not_blanc_preds)
                if len(lists['binary_proba']['train']) > 0:
                    lists['binary_proba']['train'][-1] = np.delete(lists['binary_proba']['train'][-1], train_not_blanc_preds, axis=0)
                    lists['binary_proba']['valid'][-1] = np.delete(lists['binary_proba']['valid'][-1], valid_not_blanc_preds, axis=0)
                    lists['binary_proba']['test'][-1] = np.delete(lists['binary_proba']['test'][-1], test_not_blanc_preds, axis=0)

                train_data = train_data.iloc[train_not_blanc_preds]
                valid_data = valid_data.iloc[valid_not_blanc_preds]
                test_data = test_data.iloc[test_not_blanc_preds]
                train_labels = train_labels[train_not_blanc_preds]
                valid_labels = valid_labels[valid_not_blanc_preds]
                test_labels = test_labels[test_not_blanc_preds]
                train_batches = train_batches[train_not_blanc_preds]
                valid_batches = valid_batches[valid_not_blanc_preds]
                test_batches = test_batches[test_not_blanc_preds]
                
                # Remove from train the blancs
                train_not_blancs = np.argwhere(train_labels != 'blanc').flatten()
                valid_not_blancs = np.argwhere(valid_labels != 'blanc').flatten()
                test_not_blancs = np.argwhere(test_labels != 'blanc').flatten()
                train_data = train_data.iloc[train_not_blancs]
                train_labels = train_labels[train_not_blancs]
                train_batches = train_batches[train_not_blancs]

            lists['classes']['train'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in train_labels])]
            lists['classes']['valid'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in valid_labels])]
            lists['classes']['test'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in test_labels])]
            lists['labels']['train'] += [train_labels]
            lists['labels']['valid'] += [valid_labels]
            lists['labels']['test'] += [test_labels]

            if n_aug > 0:
                train_data = augment_data(train_data, n_aug, p, g)
                train_data = pd.DataFrame(np.nan_to_num(train_data))
                train_labels = np.concatenate([train_labels] * (n_aug + 1))
                train_batches = np.concatenate([train_batches] * (n_aug + 1))
                lists['classes']['train'][-1] = np.concatenate([lists['classes']['train'][-1]] * (n_aug + 1))
            else:
                train_data = train_data.fillna(0)
            valid_data = valid_data.fillna(0)
            test_data = test_data.fillna(0)

            m = self.model()
            m.set_params(**param_grid)
            if self.args.ovr:
                m = OneVsRestClassifier(m)
            m.fit(train_data, lists['classes']['train'][-1])
            models += [m]
            try:
                # lists['acc']['train'] += [m.score(train_data, lists['classes']['train'][-1])]
                lists['preds']['train'] += [m.predict(train_data)]
            except:
                # lists['acc']['train'] += [m.score(train_data.values, lists['classes']['train'][-1])]
                lists['preds']['train'] += [m.predict(train_data.values)]

            try:
                # lists['acc']['valid'] += [m.score(valid_data, lists['classes']['valid'][-1])]
                # lists['acc']['test'] += [m.score(test_data, lists['classes']['test'][-1])]
                lists['preds']['valid'] += [m.predict(valid_data)]
                lists['preds']['test'] += [m.predict(test_data)]
            except:
                # lists['acc']['valid'] += [m.score(valid_data.values, lists['classes']['valid'][-1])]
                # lists['acc']['test'] += [m.score(test_data.values, lists['classes']['test'][-1])]
                lists['preds']['valid'] += [m.predict(valid_data.values)]
                lists['preds']['test'] += [m.predict(test_data.values)]
            try:
                lists['proba']['train'] += [m.predict_proba(train_data)]
                lists['proba']['valid'] += [m.predict_proba(valid_data)]
                lists['proba']['test'] += [m.predict_proba(test_data)]
            except:
                pass
            if self.args.remove_blancs:
                # Add a class to unique_labels, bact
                if 'bact' not in self.unique_labels:
                    self.unique_labels = np.concatenate((self.unique_labels, ['bact']))
                # # Change lists['binary_preds']['train'] blanc to blanc index and not blanc to bact index
                # train_labels = train_labels[train_not_blancs]
                # train_batches = train_batches[train_not_blancs]
                lists['bact_batches']['train'] = train_batches
                lists['bact_labels']['train'] = train_labels
                valid_data_not_blancs = valid_data.iloc[valid_not_blancs]
                lists['bact_labels']['valid'] = valid_labels[valid_not_blancs]
                lists['bact_batches']['valid'] = valid_batches[valid_not_blancs]
                test_data_not_blancs = test_data.iloc[test_not_blancs]
                lists['bact_labels']['test'] = test_labels[test_not_blancs]
                lists['bact_batches']['test'] = test_batches[test_not_blancs]
                lists['bact_classes']['train'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in lists['bact_labels']['train']])]
                lists['bact_classes']['valid'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in lists['bact_labels']['valid']])]
                lists['bact_classes']['test'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in lists['bact_labels']['test']])]

                lists['binary_preds']['train'][-1][lists['binary_preds']['train'][-1] == blanc_binary_class] = 0
                lists['binary_preds']['train'][-1][lists['binary_preds']['train'][-1] == blanc_bact_class] = len(self.unique_labels) - 1
                lists['binary_preds']['valid'][-1][lists['binary_preds']['valid'][-1] == blanc_binary_class] = 0
                lists['binary_preds']['valid'][-1][lists['binary_preds']['valid'][-1] == blanc_bact_class] = len(self.unique_labels) - 1
                lists['binary_preds']['test'][-1][lists['binary_preds']['test'][-1] == blanc_binary_class] = 0
                lists['binary_preds']['test'][-1][lists['binary_preds']['test'][-1] == blanc_bact_class] = len(self.unique_labels) - 1

                if len(lists['binary_proba']['train']) > 0:
                    lists = self.make_combined_proba(lists)

                lists['binary_classes']['train'][-1][lists['binary_classes']['train'][-1] == blanc_binary_class] = 0
                lists['binary_classes']['train'][-1][lists['binary_classes']['train'][-1] == blanc_bact_class] = len(self.unique_labels) - 1
                lists['binary_classes']['valid'][-1][lists['binary_classes']['valid'][-1] == blanc_binary_class] = 0
                lists['binary_classes']['valid'][-1][lists['binary_classes']['valid'][-1] == blanc_bact_class] = len(self.unique_labels) - 1
                lists['binary_classes']['test'][-1][lists['binary_classes']['test'][-1] == blanc_binary_class] = 0
                lists['binary_classes']['test'][-1][lists['binary_classes']['test'][-1] == blanc_bact_class] = len(self.unique_labels) - 1

                
                # TODO replace try...except with if...else
                try:
                    # lists['acc']['train'] += [m.score(train_data, lists['classes']['train'][-1])]
                    lists['bact_preds']['train'] += [m.predict(train_data)]
                except:
                    # lists['acc']['train'] += [m.score(train_data.values, lists['classes']['train'][-1])]
                    lists['bact_preds']['train'] += [m.predict(train_data.values)]

                try:
                    # lists['acc']['valid'] += [m.score(valid_data, lists['classes']['valid'][-1])]
                    # lists['acc']['test'] += [m.score(test_data, lists['classes']['test'][-1])]
                    lists['bact_preds']['valid'] += [m.predict(valid_data_not_blancs)]
                    lists['bact_preds']['test'] += [m.predict(test_data_not_blancs)]
                except:
                    # lists['acc']['valid'] += [m.score(valid_data.values, lists['classes']['valid'][-1])]
                    # lists['acc']['test'] += [m.score(test_data.values, lists['classes']['test'][-1])]
                    lists['bact_preds']['valid'] += [m.predict(valid_data_not_blancs.values)]
                    lists['bact_preds']['test'] += [m.predict(test_data_not_blancs.values)]
                try:
                    lists['bact_proba']['train'] += [m.predict_proba(train_data).argmax(1)]
                    lists['bact_proba']['valid'] += [m.predict_proba(valid_data_not_blancs).argmax(1)]
                    lists['bact_proba']['test'] += [m.predict_proba(test_data_not_blancs).argmax(1)]
                except:
                    pass
                
                lists['preds']['train'][-1] = np.concatenate([lists['preds']['train'][-1], lists['binary_preds']['train'][-1]])
                lists['preds']['valid'][-1] = np.concatenate([lists['preds']['valid'][-1], lists['binary_preds']['valid'][-1]])
                lists['preds']['test'][-1] = np.concatenate([lists['preds']['test'][-1], lists['binary_preds']['test'][-1]])
                lists['classes']['train'][-1] = np.concatenate([lists['classes']['train'][-1], lists['binary_classes']['train'][-1]])
                lists['classes']['valid'][-1] = np.concatenate([lists['classes']['valid'][-1], lists['binary_classes']['valid'][-1]])
                lists['classes']['test'][-1] = np.concatenate([lists['classes']['test'][-1], lists['binary_classes']['test'][-1]])
                lists['labels']['valid'][-1] = np.concatenate([lists['labels']['valid'][-1], lists['binary_labels']['valid'][-1]])
                lists['labels']['test'][-1] = np.concatenate([lists['labels']['test'][-1], lists['binary_labels']['test'][-1]])
                lists['labels']['train'][-1] = np.concatenate([lists['labels']['train'][-1], lists['binary_labels']['train'][-1]])
                try:
                    lists['proba']['train'][-1] = np.concatenate([lists['proba']['train'][-1], lists['binary_proba']['train'][-1]])
                    lists['proba']['valid'][-1] = np.concatenate([lists['proba']['valid'][-1], lists['binary_proba']['valid'][-1]])
                    lists['proba']['test'][-1] = np.concatenate([lists['proba']['test'][-1], lists['binary_proba']['test'][-1]])
                except:
                    pass
            else:
                blanc_class = np.argwhere(self.unique_labels == 'blanc').flatten()[0]
                blanc_binary_class = np.argwhere(unique_binary_labels == 'blanc').flatten()[0]
                train_blanc_preds = np.argwhere(lists['binary_preds']['train'][-1] == blanc_binary_class).flatten()
                valid_blanc_preds = np.argwhere(lists['binary_preds']['valid'][-1] == blanc_binary_class).flatten()
                test_blanc_preds = np.argwhere(lists['binary_preds']['test'][-1] == blanc_binary_class).flatten()
                lists['preds']['train'][-1][train_blanc_preds] = blanc_class
                lists['preds']['valid'][-1][valid_blanc_preds] = blanc_class
                lists['preds']['test'][-1][test_blanc_preds] = blanc_class

            lists['mcc']['train'] += [MCC(lists['classes']['train'][-1], lists['preds']['train'][-1])]
            lists['mcc']['valid'] += [MCC(lists['classes']['valid'][-1], lists['preds']['valid'][-1])]
            lists['mcc']['test'] += [MCC(lists['classes']['test'][-1], lists['preds']['test'][-1])]
            lists['acc']['train'] += [ACC(lists['classes']['train'][-1], lists['preds']['train'][-1])]
            lists['acc']['valid'] += [ACC(lists['classes']['valid'][-1], lists['preds']['valid'][-1])]
            lists['acc']['test'] += [ACC(lists['classes']['test'][-1], lists['preds']['test'][-1])]

            lists['bact_acc']['train'] += [ACC(lists['bact_classes']['train'][-1], lists['bact_preds']['train'][-1])]
            lists['bact_acc']['valid'] += [ACC(lists['bact_classes']['valid'][-1], lists['bact_preds']['valid'][-1])]
            lists['bact_acc']['test'] += [ACC(lists['bact_classes']['test'][-1], lists['bact_preds']['test'][-1])]
            lists['bact_mcc']['train'] += [MCC(lists['bact_classes']['train'][-1], lists['bact_preds']['train'][-1])]
            lists['bact_mcc']['valid'] += [MCC(lists['bact_classes']['valid'][-1], lists['bact_preds']['valid'][-1])]
            lists['bact_mcc']['test'] += [MCC(lists['bact_classes']['test'][-1], lists['bact_preds']['test'][-1])]

            if all_data['inputs']['urinespositives'].shape[0] > 0:
                try:
                    lists['binary_preds']['posurines'] += [
                        binary_model.predict(all_data_binary['inputs']['urinespositives'])
                    ]
                    lists['preds']['posurines'] += [m.predict(all_data['inputs']['urinespositives'])]
                    try:
                        lists['proba']['posurines'] += [m.predict_proba(all_data['inputs']['urinespositives'])]
                        lists['binary_proba']['posurines'] += [
                            binary_model.predict_proba(all_data['inputs']['urinespositives'])
                        ]
                    except:
                        pass

                except:
                    lists['preds']['posurines'] += [
                        m.predict(all_data['inputs']['urinespositives'].values)
                    ]
                    lists['binary_preds']['posurines'] += [
                        binary_model.predict(all_data_binary['inputs']['urinespositives'].values)
                    ]
                    try:
                        lists['proba']['posurines'] += [
                            m.predict_proba(all_data['inputs']['urinespositives'].values)
                        ]
                        lists['binary_proba']['posurines'] += [
                            binary_model.predict_proba(all_data['inputs']['urinespositives'].values)
                        ]
                    except:
                        pass
                # samples classified as blancs in binary model are classified as blancs in the final model
                lists['preds']['posurines'][-1][np.argwhere(lists['binary_preds']['posurines'][-1] == blanc_binary_class).flatten()] = blanc_class
                
            if self.best_scores['acc']['valid'] is None:
                self.best_scores['acc']['valid'] = 0
                # run, ae, best_lists, cols, n_meta, log_path, log_deep_only=True
            if len(lists['proba']['valid']) > 0:
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
                    },
                }
            else:
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
                    },
                }
            h += 1

        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        log_ord(self.data, uniques, ord_path, scaler_name, run)
        data = copy.deepcopy(self.data)
        metrics = log_fct(data, scaler_name, metrics)
        log_ord(data, uniques2, ord_path, f'{scaler_name}_blancs', run)

        print(lists['mcc']['valid'])
        print('valid_score:', np.mean(lists['acc']['valid']), \
              'valid_mcc:', np.mean(lists['mcc']['valid']), \
              'scaler:', scaler_name,
              'h_params:', param_grid
              )
        lists = self.save_confusion_matrices(all_data, lists, run)
        self.save_roc_curves(lists, run)
        if np.mean(lists['mcc']['valid']) > np.mean(self.best_scores['mcc']['valid']):
            
            log_shap(run, m, data_list, all_data['inputs']['all'].columns, self.bins, self.log_path)
            self.dump_models(models, lists)
            # Save the individual scores of each sample with class, #batch
            self.save_results_df(lists, run)
            self.retrieve_best_scores(lists)
            # lists = self.save_confusion_matrices(all_data, lists, run)
            # self.save_roc_curves(lists, run)

            unique_batches = {
                'train': lists['unique_batches']['train'],
                'valid': lists['unique_batches']['valid'],
                'test': lists['unique_batches']['test']
            }
            best_scores = self.save_best_model_hparams(param_grid, other_params, scaler_name, unique_batches, metrics)
        else:
            best_scores = {
                'nbe': None,
                'ari': None,
                'ami': None,
            }

        if self.args.log_neptune:
            log_neptune(run, lists, best_scores)
            run.stop()
            model.stop()

        return 1 - np.mean(lists['mcc']['valid'])


if __name__ == '__main__':
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, default='mutual_info_classif_scores.csv')
    parser.add_argument('--remove_zeros', type=int, default=0)
    parser.add_argument('--log1p', type=int, default=0)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--pool', type=int, default=0)
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='linsvc')
    parser.add_argument('--binary_model_name', type=str, default='linsvc')
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
    parser.add_argument('--shift', type=int, default=0)  # TODO keep this?
    parser.add_argument('--log', type=str, default='inloop')
    parser.add_argument('--features_selection', type=str, default='mutual_info_classif')
    parser.add_argument('--concs', type=str, default='na,h')
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument("--min_mz", type=int, default=0)
    parser.add_argument("--max_mz", type=int, default=10000)
    parser.add_argument("--min_rt", type=int, default=0)
    parser.add_argument("--max_rt", type=int, default=1000)
    parser.add_argument("--combat", type=int, default=0)
    parser.add_argument('--remove_blancs', type=int, default=1)
    args = parser.parse_args()
    args.binary = 0
    batch_dates = [
        "B13-06-05-2024", "B12-05-31-2024", "B11-05-24-2024",
        "B10-05-03-2024", "B9-04-22-2024", "B8-04-15-2024", 
        'B7-04-03-2024', 'B6-03-29-2024', 'B5-03-13-2024', 
        'B4-03-01-2024', 'B3-02-29-2024', 'B2-02-21-2024', 
        'B1-02-02-2024', 
    ]
    batches_to_keep = [
        "b13-06-05-2024", "b12-05-31-2024", "b11-05-24-2024",
        "b10-05-03-2024", "b9-04-22-2024", "b8-04-15-2024", 
        'b7-04-03-2024', 'b6-03-29-2024', 'b5-03-13-2024', 
        'b4-03-01-2024', 'b3-02-29-2024', 'b2-02-21-2024', 
        'b1-02-02-2024', 
    ]
    batch_dates = [x.split('-')[0] for x in batch_dates]
    batches_to_keep = [x.split('-')[0] for x in batches_to_keep]
    batch_date = '-'.join(batch_dates)
    args.batch_date = batch_date

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
    uniques2 = {
        'labels': None,
        'batches': unique_batches,
        'manips': unique_manips,
        'urines': unique_urines,
        'concs': unique_concs
    }

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
            Real(0, 0.5, 'uniform', name='threshold'),
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
            # Real(0, 0.5, 'uniform', name='threshold'),
            Integer(0, 5, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler"),
        ]
    elif args.model_name == 'linsvc':
        cfr = sklearn.svm.LinearSVC
        space = [
            Real(0, 0.5, 'uniform', name='threshold'),
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
            Real(0, 0.5, 'uniform', name='threshold'),
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
            Real(0, 0.5, 'uniform', name='threshold'),
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
            Real(0, 0.5, 'uniform', name='threshold'),
            Integer(0, 5, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Categorical(['minmax2', 'l2', 'l1', 'zscore'], name="scaler"),
        ]
    elif args.model_name == 'logreg':
        cfr = sklearn.linear_model.LogisticRegression
        space = [
            Real(0, 0.5, 'uniform', name='threshold'),
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
    exp_name = f'{batch_date.lower()}_2models_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_remblk{args.remove_blancs}_{cropings}_{"_".join(concs)}'
    binary_exp_name = f'{batch_date.lower()}_binary1_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_{cropings}_{"_".join(concs)}'
    path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
           f'/{args.spd}spd/thr{args.threshold}/{args.train_on}/{exp_name}/{args.model_name}/'
    binary_path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
           f'/{args.spd}spd/thr{args.threshold}/{args.train_on}/{binary_exp_name}/{args.binary_model_name}/'
    train = Train2Models(name="inputs", model=cfr, data=data, uniques=uniques, 
                  hparams_names=hparams_names, log_path=path, 
                  args=args, logger=None, 
                  log_neptune=True, mlops='None',
                  binary_path=binary_path)
        
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
