from matplotlib import rcParams, cycler

# matplotlib.use('Agg')
CUDA_VISIBLE_DEVICES = ""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import copy
import random
import json
import torch
import sklearn
import os
import pickle

from utils import scale_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings
# from datetime import datetime
# from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

from skopt import gp_minimize
from msml.utils.sklearn_train_nocv import count_labels, get_confusion_matrix, save_roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
from sklearn.multiclass import OneVsRestClassifier
from loggings import log_ord, log_fct, save_confusion_matrix
from utils import augment_data         
from dataset import get_data2 as get_data

def get_scaler_name(log_path):
    pickles = os.listdir(f'{log_path}/saved_models/')
    for x in pickles:
        if '.pkl' in x and 'scaler' in x:
            fname = x
    if 'minmax' in fname:
        return 'minmax'
    elif 'standard' in fname:
        return 'standard'
    elif 'robust' in fname:
        return 'robust'
    elif 'minmax_per_batch' in fname:
        return 'minmax_per_batch'
    elif 'standard_per_batch' in fname:
        return 'standard_per_batch'
    elif 'robust_per_batch' in fname:
        return 'robust_per_batch'
    elif 'none' in fname:
        return 'none'
    else:
        exit('WRONG SCALER NAME; not in list')

class Train:
    def __init__(self, name, model, data, uniques, hparams_names, log_path, 
                 binary_path, args, logger, ovr, groupkfold, model_name='RF', 
                 binary_model_name='linsvc', mlops='None', train_on='all_lows'):
        self.uniques = uniques
        self.groupkfold = groupkfold
        self.best_roc_score = -1
        self.ovr = ovr
        self.args = args
        self.binary_path = binary_path
        self.log_path = log_path
        self.model = model
        self.binary_model_name = binary_model_name
        self.model_name = model_name
        self.data = data
        self.logger = logger
        self.hparams_names = hparams_names
        self.train_on = train_on
        self.best_scores_train = -1
        self.best_scores_valid = -1
        self.best_mccs_train = -1
        self.best_mccs_valid = -1
        self.scores_train = None
        self.scores_valid = None
        self.mccs_train = None
        self.mccs_valid = None
        self.y_preds = np.array([])
        self.y_valids = np.array([])
        self.iter = 0
        self.model = model
        self.name = name
        self.mlops = mlops
        self.best_params_dict = {}
        self.best_params_dict_values = {}

    def train(self, h_params):
        metrics = {}
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        scaler_name = 'none'
        for name, param in zip(self.hparams_names, h_params):
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

        other_params = {'p': p, 'g': g, 'n_aug': n_aug}
        scaler_name = get_scaler_name(scaler_name)
        scaler = pickle.load(open(f'{self.log_path}/saved_models/{scaler_name}_scaler.pkl', 'rb'))
        self.data = scaler()
        
        scores_valid = []
        scores_train = []
        scores_test = []
        y_preds_train = []
        y_preds_valid = []
        y_preds_test = []
        y_binary_preds_train = []
        y_binary_preds_valid = []
        y_binary_preds_test = []

        train_classes = []
        valid_classes = []
        test_classes = []
        
        train_binary_classes = []
        valid_binary_classes = []
        test_binary_classes = []
        
        mccs_train = []
        mccs_valid = []
        mccs_test = []
        all_data = self.data['inputs']['all']
        all_labels = self.data['labels']['all']
        all_batches = self.data['batches']['all']
        all_concs = self.data['concs']['all']

        # place blancs at the end
        unique_labels = np.array(np.unique(all_labels))
        blanc_class = np.argwhere(unique_labels == 'blanc').flatten()[0]
        unique_labels = np.concatenate((['blanc'], np.delete(unique_labels, blanc_class)))
        
        all_binary_labels = pd.Series(['blanc' if label == 'blanc' else 'bact' for label in all_labels])
        unique_binary_labels = np.array(np.unique(all_binary_labels))
        unique_binary_labels = np.concatenate((['blanc'], np.delete(unique_binary_labels, np.argwhere(unique_binary_labels == 'blanc').flatten()[0])))

        print(f'Iteration: {self.iter}')
        combinations = []
        models = []
        h = 0
        seed = 0
        print(unique_labels)
        train_batches_list = []
        valid_batches_list = []
        test_batches_list = []

        while h < self.args.n_repeats:
            # import train valid and test indices
            with open(f'{self.binary_path}/saved_models/{self.args.binary_model_name}_{h}_train_indices.pkl', 'rb') as f:
                train_inds = pickle.load(f)
            with open(f'{self.binary_path}/saved_models/{self.binary_model_name}_{h}_valid_indices.pkl', 'rb') as f:
                valid_inds = pickle.load(f)
            with open(f'{self.binary_path}/saved_models/{self.binary_model_name}_{h}_test_indices.pkl', 'rb') as f:
                test_inds = pickle.load(f)
            # import binary best model
            with open(f'{self.binary_path}/saved_models/{self.binary_model_name}_{h}.pkl', 'rb') as f:
                binary_model = pickle.load(f)

            train_data, valid_data, test_data = all_data.iloc[train_inds], all_data.iloc[valid_inds], all_data.iloc[test_inds]
            train_labels, valid_labels, test_labels = all_labels[train_inds], all_labels[valid_inds], all_labels[test_inds]
            train_batches, valid_batches, test_batches = all_batches[train_inds], all_batches[valid_inds], all_batches[test_inds]

            train_batches_list += [list(np.unique(train_batches))]
            valid_batches_list += [list(np.unique(valid_batches))]
            test_batches_list += [list(np.unique(test_batches))]
            unique_train_labels = np.array(np.unique(train_labels))
            # find which label is missing in train
            # which of unique_labels is not in unique_train_labels
            not_in_train_labels = np.array([l for l in unique_labels if l not in np.unique(train_labels)])

            # not_in_valid_labels = np.array([l for l in unique_labels if l not in np.unique(valid_labels)])
            # not_in_test_labels = np.array([l for l in unique_labels if l not in np.unique(test_labels)])

            # Remove labels from train that are missing from unique_labels
            train_inds = np.argwhere(np.isin(train_labels, unique_train_labels)).flatten()
            valid_inds = np.argwhere(np.isin(valid_labels, unique_train_labels)).flatten()
            test_inds = np.argwhere(np.isin(test_labels, unique_train_labels)).flatten()
            train_labels = train_labels[train_inds]
            valid_labels = valid_labels[valid_inds]
            test_labels = test_labels[test_inds]
            train_batches = train_batches[train_inds]
            valid_batches = valid_batches[valid_inds]
            test_batches = test_batches[test_inds]
            train_data = train_data.iloc[train_inds]
            valid_data = valid_data.iloc[valid_inds]
            test_data = test_data.iloc[test_inds]

            train_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in train_labels])
            valid_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in valid_labels])
            test_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in test_labels])

            train_binary_classes += [np.array([np.argwhere(l == unique_binary_labels).flatten() for l in train_binary_labels])]
            valid_binary_classes += [np.array([np.argwhere(l == unique_binary_labels).flatten() for l in valid_binary_labels])]
            test_binary_classes += [np.array([np.argwhere(l == unique_binary_labels).flatten() for l in test_binary_labels])]

            y_binary_preds_train += [binary_model.predict(train_data)]
            y_binary_preds_valid += [binary_model.predict(valid_data)]
            y_binary_preds_test += [binary_model.predict(test_data)]

            if self.args.remove_blancs:
                blanc_class = np.argwhere(unique_labels == 'blanc').flatten()[0]
                # drop blanc class in unique_labels
                blanc_binary_class = np.argwhere(unique_binary_labels == 'blanc').flatten()[0]
                blanc_bact_class = np.argwhere(unique_binary_labels != 'blanc').flatten()[0]
                train_not_blanc_preds = np.argwhere(y_binary_preds_train[-1] != blanc_binary_class).flatten()
                valid_not_blanc_preds = np.argwhere(y_binary_preds_valid[-1] != blanc_binary_class).flatten()
                test_not_blanc_preds = np.argwhere(y_binary_preds_test[-1] != blanc_binary_class).flatten()

                # keep only predictions that are blancs
                train_binary_classes[-1] = np.delete(train_binary_classes[-1], train_not_blanc_preds)
                valid_binary_classes[-1] = np.delete(valid_binary_classes[-1], valid_not_blanc_preds)
                test_binary_classes[-1] = np.delete(test_binary_classes[-1], test_not_blanc_preds)
                y_binary_preds_train[-1] = np.delete(y_binary_preds_train[-1], train_not_blanc_preds)
                y_binary_preds_valid[-1] = np.delete(y_binary_preds_valid[-1], valid_not_blanc_preds)
                y_binary_preds_test[-1] = np.delete(y_binary_preds_test[-1], test_not_blanc_preds)
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
                train_data = train_data.iloc[train_not_blancs]
                train_labels = train_labels[train_not_blancs]
                train_batches = train_batches[train_not_blancs]

            train_classes += [np.array([np.argwhere(l == unique_labels)[0][0] for l in train_labels])]
            valid_classes += [np.array([np.argwhere(l == unique_labels)[0][0] for l in valid_labels])]
            test_classes += [np.array([np.argwhere(l == unique_labels)[0][0] for l in test_labels])]
            # remove samples which are predicted as blancs
            m = self.model()
            m.set_params(**param_grid)
            if self.ovr:
                m = OneVsRestClassifier(m)

            if n_aug > 0:
                train_data = augment_data(train_data, n_aug, p, g)
                train_classes[-1] = np.concatenate([train_classes[-1]] * (n_aug + 1))
                # train_batches = np.concatenate([train_batches] * (n_aug + 1))

            if train_data.shape[0] == 0 or valid_data.shape[0] == 0 or test_data.shape[0] == 0:
                return 1
            m.fit(train_data, train_classes[-1])
            models += [m]

            y_preds_train += [m.predict(train_data)]
            y_preds_valid += [m.predict(valid_data)]
            y_preds_test += [m.predict(test_data)]
            
            if self.args.remove_blancs:
                # Add a class to unique_labels, bact
                if 'bact' not in unique_labels:
                    unique_labels = np.concatenate((unique_labels, ['bact']))
                # # Change y_binary_preds_train blanc to blanc index and not blanc to bact index
                y_binary_preds_train[-1][y_binary_preds_train[-1] == blanc_binary_class] = 0
                y_binary_preds_train[-1][y_binary_preds_train[-1] == blanc_bact_class] = len(unique_labels) - 1
                y_binary_preds_valid[-1][y_binary_preds_valid[-1] == blanc_binary_class] = 0
                y_binary_preds_valid[-1][y_binary_preds_valid[-1] == blanc_bact_class] = len(unique_labels) - 1
                y_binary_preds_test[-1][y_binary_preds_test[-1] == blanc_binary_class] = 0
                y_binary_preds_test[-1][y_binary_preds_test[-1] == blanc_bact_class] = len(unique_labels) - 1
                train_binary_classes[-1][train_binary_classes[-1] == blanc_binary_class] = 0
                train_binary_classes[-1][train_binary_classes[-1] == blanc_bact_class] = len(unique_labels) - 1
                valid_binary_classes[-1][valid_binary_classes[-1] == blanc_binary_class] = 0
                valid_binary_classes[-1][valid_binary_classes[-1] == blanc_bact_class] = len(unique_labels) - 1
                test_binary_classes[-1][test_binary_classes[-1] == blanc_binary_class] = 0
                test_binary_classes[-1][test_binary_classes[-1] == blanc_bact_class] = len(unique_labels) - 1

                y_preds_train[-1] = np.concatenate([y_preds_train[-1], y_binary_preds_train[-1]])
                y_preds_valid[-1] = np.concatenate([y_preds_valid[-1], y_binary_preds_valid[-1]])
                y_preds_test[-1] = np.concatenate([y_preds_test[-1], y_binary_preds_test[-1]])
                train_classes[-1] = np.concatenate([train_classes[-1], train_binary_classes[-1]])
                valid_classes[-1] = np.concatenate([valid_classes[-1], valid_binary_classes[-1]])
                test_classes[-1] = np.concatenate([test_classes[-1], test_binary_classes[-1]])
            else:
                blanc_class = np.argwhere(unique_labels == 'blanc').flatten()[0]
                blanc_binary_class = np.argwhere(unique_binary_labels == 'blanc').flatten()[0]
                train_blanc_preds = np.argwhere(y_binary_preds_train[-1] == blanc_binary_class).flatten()
                valid_blanc_preds = np.argwhere(y_binary_preds_valid[-1] == blanc_binary_class).flatten()
                test_blanc_preds = np.argwhere(y_binary_preds_test[-1] == blanc_binary_class).flatten()
                y_preds_train[-1][train_blanc_preds] = blanc_class
                y_preds_valid[-1][valid_blanc_preds] = blanc_class
                y_preds_test[-1][test_blanc_preds] = blanc_class

            mccs_train += [MCC(train_classes[-1], y_preds_train[-1])]
            mccs_valid += [MCC(valid_classes[-1], y_preds_valid[-1])]
            mccs_test += [MCC(test_classes[-1], y_preds_test[-1])]
            scores_train += [ACC(train_classes[-1], y_preds_train[-1])]
            scores_valid += [ACC(valid_classes[-1], y_preds_valid[-1])]
            scores_test += [ACC(test_classes[-1], y_preds_test[-1])]
            if self.best_scores_valid is None:
                self.best_scores_valid = 0
            h += 1

        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        log_ord(self.data, self.uniques['labels'], self.uniques['batches'], 
                self.uniques['manips'], self.uniques['urines'], self.uniques['concs'], ord_path, scaler_name)
        data = copy.deepcopy(self.data)
        metrics = log_fct(data, scaler_name, metrics)
        log_ord(data, None, self.uniques['batches'], 
                self.uniques['manips'], self.uniques['urines'],
                self.uniques['concs'], ord_path, 
                f'{scaler_name}_blancs')

        print(mccs_valid)
        print('valid_score:', np.mean(scores_valid), \
              'valid_mcc:', np.mean(mccs_valid), \
              'scaler:', scaler_name,
              'h_params:', param_grid
              )
        if np.mean(scores_valid) > np.mean(self.best_scores_valid):
            for i, m in enumerate(models):
                # save model
                os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
                with open(f'{self.log_path}/saved_models/{self.name}_{self.model_name}_{i}.pkl', 'wb') as f:
                    pickle.dump(m, f)
            self.best_scores_train = scores_train
            self.best_scores_valid = scores_valid
            self.best_scores_test = scores_test
            self.best_mccs_train = mccs_train
            self.best_mccs_valid = mccs_valid
            self.best_mccs_test = mccs_test
            unique_train_preds = np.unique(np.concatenate(y_preds_train))
            unique_valid_preds = np.unique(np.concatenate(y_preds_valid))
            unique_test_preds = np.unique(np.concatenate(y_preds_test))
            unique_train_preds = np.concatenate([np.array([unique_labels[int(i)] for i in unique_train_preds]), np.array(['bact'])])
            unique_valid_preds = np.concatenate([np.array([unique_labels[int(i)] for i in unique_valid_preds]), np.array(['bact'])])
            unique_test_preds = np.concatenate([np.array([unique_labels[int(i)] for i in unique_test_preds]), np.array(['bact'])])

            fig = get_confusion_matrix(np.concatenate(train_classes), 
                                       np.concatenate(y_preds_train), 
                                       unique_train_preds)
            save_confusion_matrix(
                fig, 
                f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_train",
                acc=scores_train, mcc=mccs_train, group='train'
            )
            fig = get_confusion_matrix(np.concatenate(valid_classes), 
                                       np.concatenate(y_preds_valid), 
                                       unique_valid_preds)
            save_confusion_matrix(
                fig, 
                f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_valid",
                acc=scores_valid, mcc=mccs_valid, group='valid'
            )
            fig = get_confusion_matrix(np.concatenate(test_classes), 
                                       np.concatenate(y_preds_test), 
                                       unique_test_preds)
            save_confusion_matrix(
                fig, 
                f"{self.log_path}/confusion_matrices/{self.name}_{self.model_name}_test", 
                acc=scores_test, mcc=mccs_test, group='test'
            )
            try:
                self.best_roc_train = save_roc_curve(m, train_data, train_classes, unique_labels,
                                                     f"{self.log_path}/ROC/{self.name}_{self.model_name}_train", binary=0,
                                                     acc=score_train)
            except:
                pass
            try:
                self.best_roc_valid = save_roc_curve(m, valid_data, valid_classes, unique_labels,
                                                     f"{self.log_path}/ROC/{self.name}_{self.model_name}_valid", binary=0,
                                                     acc=score_valid)
            except:
                pass
            try:
                self.best_roc_test = save_roc_curve(m, test_data, test_classes, unique_labels,
                                                     f"{self.log_path}/ROC/{self.name}_{self.model_name}_test", binary=0,
                                                     acc=score_test)
            except:
                pass
            unique_batches = {
                'train': train_batches_list,
                'valid': valid_batches_list,
                'test': test_batches_list
            }
            self.save_best_model_hparams(param_grid, other_params, scaler_name, unique_batches, metrics)

        return 1 - np.mean(scores_valid)

    def save_best_model_hparams(self, params, other_params, scaler_name, unique_batches, metrics):
        with open(f'{self.log_path}/saved_models/unique_batches_{self.name}_{self.model_name}.json', "w") as read_file:
            json.dump(unique_batches, read_file)
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

        with open(f'{self.log_path}/saved_models/unique_batches_{self.name}_{self.model_name}.json', "w") as read_file:
            json.dump(unique_batches, read_file)

        self.best_params_dict_values['train_batches'] = self.best_scores_train

        self.best_params_dict_values['train_acc'] = self.best_scores_train
        self.best_params_dict_values['valid_acc'] = self.best_scores_valid
        self.best_params_dict_values['test_acc'] = self.best_scores_test
        self.best_params_dict_values['train_acc'] = self.best_scores_train
        self.best_params_dict_values['valid_acc'] = self.best_scores_valid
        self.best_params_dict_values['test_acc'] = self.best_scores_test

        self.best_params_dict_values['train_mcc'] = self.best_mccs_train
        self.best_params_dict_values['valid_mcc'] = self.best_mccs_valid
        self.best_params_dict_values['test_mcc'] = self.best_mccs_test
        self.best_params_dict_values['train_mcc'] = self.best_mccs_train
        self.best_params_dict_values['valid_mcc'] = self.best_mccs_valid
        self.best_params_dict_values['test_mcc'] = self.best_mccs_test
        
        self.best_params_dict['ami'] = metrics[scaler_name]['all']['adjusted_mutual_info_score']['domains']
        self.best_params_dict['ari'] = metrics[scaler_name]['all']['adjusted_rand_score']['domains']
        n_batches = np.unique(np.concatenate([np.concatenate(unique_batches[b]) for b in unique_batches])).flatten().shape[0]
        self.best_params_dict['nBE'] = (np.log(n_batches) - metrics[scaler_name]['all']['shannon']['domains'])  / np.log(n_batches)

        self.best_params_dict['train_acc_mean'] = np.mean(self.best_scores_train)
        self.best_params_dict['valid_acc_mean'] = np.mean(self.best_scores_valid)
        self.best_params_dict['test_acc_mean'] = np.mean(self.best_scores_test)
        self.best_params_dict['train_acc_std'] = np.std(self.best_scores_train)
        self.best_params_dict['valid_acc_std'] = np.std(self.best_scores_valid)
        self.best_params_dict['test_acc_std'] = np.std(self.best_scores_test)

        self.best_params_dict['train_mcc_mean'] = np.mean(self.best_mccs_train)
        self.best_params_dict['valid_mcc_mean'] = np.mean(self.best_mccs_valid)
        self.best_params_dict['test_mcc_mean'] = np.mean(self.best_mccs_test)
        self.best_params_dict['train_mcc_std'] = np.std(self.best_mccs_train)
        self.best_params_dict['valid_mcc_std'] = np.std(self.best_mccs_valid)
        self.best_params_dict['test_mcc_std'] = np.std(self.best_mccs_test)

        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.model_name}.json', "w") as read_file:
            json.dump(self.best_params_dict, read_file)
        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.model_name}_values.json', "w") as read_file:
            json.dump(self.best_params_dict_values, read_file)
        # load model
                   

# def main
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
    # TODO also include binary_n_features to allow not the same features
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--binary_model_name', type=str, default='linsvc')
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
    parser.add_argument('--remove_blancs', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=10)
    args = parser.parse_args()

    batch_dates = [
        'B1-02-02-2024', 'B2-02-21-2024', 'B3-02-29-2024', 
        'B4-03-01-2024', 'B5-03-13-2024', 'B6-03-29-2024',
        'B7-04-03-2024', "B8-04-15-2024", "B9-04-22-2024"
        ]
    batches_to_keep = [
        'b1-02-02-2024', 'b2-02-21-2024', 'b3-02-29-2024',
        'b4-03-01-2024', 'b5-03-13-2024', 'b6-03-29-2024',
        'b7-04-03-2024', 'b8-04-15-2024', 'b9-04-22-2024'
        ]
    batch_date = '-'.join(batch_dates)
    concs = args.concs.split(',')
    # exp_name = f'{batch_date}_binary{args.binary}_{args.n_features}' \
    #                     f'_gkf{args.groupkfold}_ovr{args.ovr}_{"_".join(concs)}'

    # TODO change gkf0; only valid because using all features
    exp = f'all_{batch_date}_gkf0_5splits'  
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

    from collections import Counter

    n_cats = len(np.unique(data['labels']['all']))
    n_batches = len(np.unique(data['batches']['all']))

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
    
    print(Counter(data['labels']['all']))
    if args.model_name == 'RF':
        space = [
            # Integer(100, 20000, 'uniform', name='features_cutoff'),
            # Real(0, 1, 'uniform', name='threshold'),
            Integer(1, 5, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(1, 100, 'uniform', name="max_features"),
            Integer(2, 10, 'uniform', name="min_samples_split"),
            Integer(1, 10, 'uniform', name="min_samples_leaf"),
            Integer(1, 1000, 'uniform', name="n_estimators"),
            Categorical(['gini', 'entropy'], name="criterion"),
            Categorical([True, False], name="oob_score"),
            Categorical(['balanced'], name="class_weight"),
            Categorical(['minmax'], name="scaler"),
        ]
    elif args.model_name == 'linsvc':
        space = [
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Real(1e-4, 2, 'log-uniform', name='tol'),
            Integer(1, 10000, 'uniform', name='max_iter'),
            Categorical(['l2'], name='penalty'),
            Categorical(['hinge', 'squared_hinge'], name='loss'),
            Real(1e-3, 10000, 'uniform', name='C'),
            Categorical(['balanced'], name='class_weight'),
            Categorical(['robust'], name="scaler")
        ]
    elif args.model_name == 'LDA':
        space = [
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Categorical(['minmax', 'standard', 'robust', 
                         'minmax_per_batch', 'standard_per_batch', 
                         'robust_per_batch'], name="scaler"),
        ]
    elif args.model_name == 'logreg':
        space = [
            Integer(1, 20000, 'uniform', name='max_iter'),
            Real(1e-3, 20000, 'uniform', name='C'),
            Categorical(['saga'], name='solver'),
            Categorical(['l1', 'l2'], name='penalty'),
            Categorical([True, False], name='fit_intercept'),
            Categorical(['balanced'], name='class_weight'),
            Categorical(['minmax', 'standard', 'robust', 
                         'minmax_per_batch', 'standard_per_batch', 
                         'robust_per_batch'], name="scaler"),
        ]
    
    if args.model_name == 'RF':
        cfr = RandomForestClassifier
    elif args.model_name == 'linsvc':
        cfr = sklearn.svm.LinearSVC
    elif args.model_name == 'logreg':
        cfr = sklearn.linear_model.LogisticRegression
    elif args.model_name == 'LDA':
        cfr = LDA


    hparams_names = [x.name for x in space]
    exp_name = f'{batch_date}_2models_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_remblk{args.remove_blancs}_{"_".join(concs)}'
    binary_exp_name = f'{batch_date}_binary1_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_{"_".join(concs)}'
    path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
           f'{args.train_on}/{exp_name}/{args.model_name}/'
    binary_path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
           f'{args.train_on}/{binary_exp_name}/{args.binary_model_name}/'
    train = Train(name="inputs", model=cfr, data=data, uniques=uniques, 
                  hparams_names=hparams_names, log_path=path, 
                  binary_path=binary_path, args=args, logger=None, 
                  ovr=args.ovr, groupkfold=args.groupkfold,
                  mlops='None', model_name=args.model_name, 
                  binary_model_name=args.binary_model_name, 
                  train_on=args.train_on)
    
    res = gp_minimize(train.train, space, n_calls=20, random_state=1)
