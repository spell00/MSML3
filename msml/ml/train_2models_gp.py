import pandas as pd
import numpy as np
import random
import json
import sklearn.neighbors
import torch
import sklearn
import os
import pickle

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from utils import scale_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

warnings.filterwarnings("ignore")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
import copy

from skopt import gp_minimize
from sklearn_train_nocv import count_labels, get_confusion_matrix, save_roc_curve, plot_roc
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
from sklearn.multiclass import OneVsRestClassifier
# from sklearn import metrics
from loggings import log_ord, log_fct, save_confusion_matrix
from utils import augment_data         
from dataset import get_data

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def classify_posurines(y_preds_posurines, unique_labels):
    posurines_classes = []
    for preds in y_preds_posurines:
        posurines_classes += [np.array([np.argwhere(l == unique_labels)[0][0] for l in preds])]
    return posurines_classes


class Train:
    def __init__(self, name, model, data, uniques, hparams_names, log_path, 
                 binary_path, args, logger, mlops='None'):

        self.binary_path = binary_path
        self.best_roc_score = -1
        self.args = args
        self.log_path = log_path
        self.model = model
        self.data = data
        self.logger = logger
        self.hparams_names = hparams_names
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
        self.uniques = uniques
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

        scores_valid = []
        scores_train = []
        scores_test = []
        y_preds_train = []
        y_preds_valid = []
        y_preds_test = []
        y_preds_posurines = []
        y_preds_posurines_binary = []
        y_proba_train = []
        y_proba_valid = []
        y_proba_test = []
        y_proba_posurines = []
        y_proba_posurines_binary = []

        train_classes = []
        valid_classes = []
        test_classes = []

        train_labels_list = []
        valid_labels_list = []
        test_labels_list = []
        train_binary_labels_list = []
        valid_binary_labels_list = []
        test_binary_labels_list = []

        train_binary_classes = []
        valid_binary_classes = []
        test_binary_classes = []
        y_binary_preds_train = []
        y_binary_preds_valid = []
        y_binary_preds_test = []

        mccs_train = []
        mccs_valid = []
        mccs_test = []
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
        unique_labels = np.array(np.unique(all_data['labels']['all']))
        # place blancs at the end
        blanc_class = np.argwhere(unique_labels == 'blanc').flatten()[0]
        unique_labels = np.concatenate((['blanc'], np.delete(unique_labels, blanc_class)))
       
        all_binary_labels = pd.Series(['blanc' if label == 'blanc' else 'bact' for label in all_data['labels']['all']])	
        unique_binary_labels = np.array(np.unique(all_binary_labels))
        unique_binary_labels = np.concatenate((['blanc'], np.delete(unique_binary_labels, np.argwhere(unique_binary_labels == 'blanc').flatten()[0])))

        # NORMALIZE DATA
        all_data, scaler = scale_data(scaler_name, all_data)
        os.makedirs(f'{self.log_path}/saved_models/', exist_ok=True)
        with open(f'{self.log_path}/saved_models/{scaler_name}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        # NORMALIZE BINARY DATA
        # import binary model scaler
        json_file = f'{self.binary_path}/saved_models/best_params_inputs_{self.args.binary_model_name}.json'
        with open(json_file, 'r') as f:
            binary_params = json.load(f)
            bscaler_name = binary_params['scaler']
        with open(f'{self.binary_path}/saved_models/{bscaler_name}_scaler.pkl', 'rb') as f:
            bscaler = pickle.load(f)

        if bscaler_name == 'minmax2':
            all_data, scaler = scale_data(scaler_name, all_data)
        else:
            all_data_binary['inputs']['all'].iloc[:] = bscaler.transform(all_data_binary['inputs']['all'])
            all_data_binary['inputs']['train'].iloc[:] = bscaler.transform(all_data_binary['inputs']['train'])
            all_data_binary['inputs']['valid'].iloc[:] = bscaler.transform(all_data_binary['inputs']['valid'])
            all_data_binary['inputs']['test'].iloc[:] = bscaler.transform(all_data_binary['inputs']['test'])
        
        print(f'Iteration: {self.iter}')
        combinations = []
        models = []
        h = 0
        seed = 0
        train_inds_list = []
        valid_inds_list = []
        test_inds_list = []
        train_batches_list = []
        valid_batches_list = []
        test_batches_list = []
        train_unique_batches_list = []
        valid_unique_batches_list = []
        test_unique_batches_list = []
        if self.args.groupkfold:
            self.args.n_repeats = len(np.unique(all_data['batches']['all']))
        while h < self.args.n_repeats:
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
            
            train_data, valid_data, test_data = all_data['inputs']['all'].iloc[train_inds], all_data['inputs']['all'].iloc[valid_inds], all_data['inputs']['all'].iloc[test_inds]
            train_labels, valid_labels, test_labels = all_data['labels']['all'][train_inds], all_data['labels']['all'][valid_inds], all_data['labels']['all'][test_inds]
            train_batches, valid_batches, test_batches = all_data['batches']['all'][train_inds], all_data['batches']['all'][valid_inds], all_data['batches']['all'][test_inds]
            train_binary_data, valid_binary_data, test_binary_data = all_data_binary['inputs']['all'].iloc[train_inds], all_data_binary['inputs']['all'].iloc[valid_inds], all_data_binary['inputs']['all'].iloc[test_inds]
            # train_binary_labels, valid_binary_labels, test_binary_labels = all_binary_data['labels']['all'][train_inds], all_binary_data['labels']['all'][valid_inds], all_binary_data['labels']['all'][test_inds]
            # train_binary_batches, valid_binary_batches, test_binary_batches = all_binary_data['batches']['all'][train_inds], all_binary_data['batches']['all'][valid_inds], all_binary_data['batches']['all'][test_inds]

            train_inds_list += [train_inds]
            valid_inds_list += [valid_inds]
            test_inds_list += [test_inds]

            train_batches_list += [train_batches]
            valid_batches_list += [valid_batches]
            test_batches_list += [test_batches]
            train_unique_batches_list += [list(np.unique(train_batches))]
            valid_unique_batches_list += [list(np.unique(valid_batches))]
            test_unique_batches_list += [list(np.unique(test_batches))]

            train_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in train_labels])
            valid_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in valid_labels])
            test_binary_labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in test_labels])
            train_binary_labels_list += [train_binary_labels]
            valid_binary_labels_list += [valid_binary_labels]
            test_binary_labels_list += [test_binary_labels]

            train_binary_classes += [
                np.array(
                    [np.argwhere(l == unique_binary_labels).flatten() for l in train_binary_labels]
                )
            ]
            valid_binary_classes += [
                np.array(
                    [np.argwhere(l == unique_binary_labels).flatten() for l in valid_binary_labels]
                )
            ]
            test_binary_classes += [
                np.array(
                    [np.argwhere(l == unique_binary_labels).flatten() for l in test_binary_labels]
                    )
            ]

            y_binary_preds_train += [binary_model.predict(train_binary_data)]
            y_binary_preds_valid += [binary_model.predict(valid_binary_data)]
            y_binary_preds_test += [binary_model.predict(test_binary_data)]

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
                train_binary_labels_list[-1] = np.delete(train_binary_labels_list[-1].to_numpy(), train_not_blanc_preds)
                valid_binary_labels_list[-1] = np.delete(valid_binary_labels_list[-1].to_numpy(), valid_not_blanc_preds)
                test_binary_labels_list[-1] = np.delete(test_binary_labels_list[-1].to_numpy(), test_not_blanc_preds)

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
            train_labels_list += [train_labels]
            valid_labels_list += [valid_labels]
            test_labels_list += [test_labels]

            if n_aug > 0:
                train_data = augment_data(train_data, n_aug, p, g)
                train_data = pd.DataFrame(np.nan_to_num(train_data))
                train_labels = np.concatenate([train_labels] * (n_aug + 1))
                train_batches = np.concatenate([train_batches] * (n_aug + 1))
                train_classes[-1] = np.concatenate([train_classes[-1]] * (n_aug + 1))
            else:
                train_data = train_data.fillna(0)
            valid_data = valid_data.fillna(0)
            test_data = test_data.fillna(0)

            m = self.model()
            m.set_params(**param_grid)
            if self.args.ovr:
                m = OneVsRestClassifier(m)
            m.fit(train_data, train_classes[-1])
            models += [m]
            try:
                # scores_train += [m.score(train_data, train_classes[-1])]
                y_preds_train += [m.predict(train_data)]
            except:
                # scores_train += [m.score(train_data.values, train_classes[-1])]
                y_preds_train += [m.predict(train_data.values)]

            try:
                # scores_valid += [m.score(valid_data, valid_classes[-1])]
                # scores_test += [m.score(test_data, test_classes[-1])]
                y_preds_valid += [m.predict(valid_data)]
                y_preds_test += [m.predict(test_data)]
            except:
                # scores_valid += [m.score(valid_data.values, valid_classes[-1])]
                # scores_test += [m.score(test_data.values, test_classes[-1])]
                y_preds_valid += [m.predict(valid_data.values)]
                y_preds_test += [m.predict(test_data.values)]
            try:
                y_proba_train += [m.predict_proba(train_data).argmax(1)]
                y_proba_valid += [m.predict_proba(valid_data).argmax(1)]
                y_proba_test += [m.predict_proba(test_data).argmax(1)]
            except:
                pass
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
                valid_labels_list[-1] = np.concatenate([valid_labels_list[-1], valid_binary_labels_list[-1]])
                test_labels_list[-1] = np.concatenate([test_labels_list[-1], test_binary_labels_list[-1]])
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
            scores_train += [MCC(train_classes[-1], y_preds_train[-1])]
            scores_valid += [MCC(valid_classes[-1], y_preds_valid[-1])]
            scores_test += [MCC(test_classes[-1], y_preds_test[-1])]

            if all_data['inputs']['urinespositives'].shape[0] > 0:
                try:
                    y_preds_posurines_binary += [binary_model.predict(all_data['inputs']['urinespositives'])]
                    y_preds_posurines += [m.predict(all_data['inputs']['urinespositives'])]
                    try:
                        y_proba_posurines += [m.predict_proba(all_data['inputs']['urinespositives'])]
                        y_proba_posurines_binary += [binary_model.predict_proba(all_data['inputs']['urinespositives'])]
                    except:
                        pass

                except:
                    y_preds_posurines += [m.predict(all_data['inputs']['urinespositives'].values)]
                    y_preds_posurines_binary += [binary_model.predict(all_data['inputs']['urinespositives'].values)]
                    try:
                        y_proba_posurines += [m.predict_proba(all_data['inputs']['urinespositives'].values)]
                        y_proba_posurines_binary += [binary_model.predict_proba(all_data['inputs']['urinespositives'].values)]
                    except:
                        pass
                # samples classified as blancs in binary model are classified as blancs in the final model
                y_preds_posurines[-1][np.argwhere(y_preds_posurines_binary[-1] == blanc_binary_class).flatten()] = blanc_class
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
        if np.mean(mccs_valid) > np.mean(self.best_mccs_valid):
            for i, m in enumerate(models):
                # save model
                with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}.pkl', 'wb') as f:
                    pickle.dump(m, f)
                # save indices
                with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}_train_indices.pkl', 'wb') as f:
                    pickle.dump(train_inds_list[i], f)
                with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}_valid_indices.pkl', 'wb') as f:
                    pickle.dump(valid_inds_list[i], f)
                with open(f'{self.log_path}/saved_models/{self.args.model_name}_{i}_test_indices.pkl', 'wb') as f:
                    pickle.dump(test_inds_list[i], f)
            # Save the individual scores of each sample with class, #batch
            if len(y_proba_train) > 0:
                df_valid = pd.DataFrame(
                    {
                        'classes': np.concatenate(valid_classes),
                        'labels': np.concatenate(valid_labels_list),
                        'batches': np.concatenate(valid_batches_list),
                        'preds': np.concatenate(y_preds_valid), 
                        'proba': np.concatenate(y_proba_valid)                   
                    }
                )
                df_test = pd.DataFrame(
                    {
                        'classes': np.concatenate(test_classes),
                        'labels': np.concatenate(test_labels_list),
                        'batches': np.concatenate(test_batches_list),
                        'preds': np.concatenate(y_preds_test),
                        'proba': np.concatenate(y_proba_test)                   
                    }
                )

            else:
                df_valid = pd.DataFrame(
                    {
                        'classes': np.concatenate(valid_classes),
                        'labels': np.concatenate(valid_labels_list),
                        'batches': np.concatenate(valid_batches_list),
                        'preds': np.concatenate(y_preds_valid),                    
                    }
                )
                df_test = pd.DataFrame(
                    {
                        'classes': np.concatenate(test_classes),
                        'labels': np.concatenate(test_labels_list),
                        'batches': np.concatenate(test_batches_list),
                        'preds': np.concatenate(y_preds_test),                    
                    }
                )
                    
            df_valid.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_valid_individual_results.csv')
            df_test.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_test_individual_results.csv')

            urinespositives_names = np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])
            urinespositives_batches = np.array([x for x in all_data['batches']['urinespositives']])
            # TODO load expected classes elsewhere
            urinespositives_real_df = pd.read_csv(f'resources/bacteries_2024/B10-05-03-2024/b10_patients_samples.csv')
            urinespositives_real_df.loc[:, 'Class'] = [l.lower() for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]
            urinespositives_classes = [int(np.argwhere(l == unique_labels).flatten()) for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]
            
            # remove samples from urinespositives_names that are not in urinespositives_real_df.loc[:, 'ID']
            to_keep = np.argwhere(np.isin(urinespositives_names, urinespositives_real_df.loc[:, 'ID'].to_numpy()) == True).flatten()
            urinespositives_names = urinespositives_names[to_keep]
            urinespositives_batches = urinespositives_batches[to_keep]
            y_preds_posurines = [[y_preds_posurines[j][i] for i in to_keep] for j in range(len(y_preds_posurines))]
            # make the order of  urinespositives_real_df.loc[:, 'Class'] the same as urinespositives_names
            new_order = np.argsort(urinespositives_names)
            urinespositives_real_df = urinespositives_real_df.iloc[new_order]
            # make sure the order is the same
            assert np.sum([x == urinespositives_real_df.loc[:, 'ID'].to_numpy() for x in urinespositives_names]) == len(urinespositives_names)
            try:
                posurines_df = pd.DataFrame(
                    {
                        'names': np.concatenate([urinespositives_names for _ in range(len(y_preds_posurines))]),
                        'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(y_preds_posurines)]),
                        'batches': np.concatenate([urinespositives_batches for _ in range(len(y_preds_posurines))]),
                        'preds': np.concatenate(y_preds_posurines),
                        'labels': np.concatenate([urinespositives_real_df.loc[:, 'Class'] for _ in range(len(y_preds_posurines))]),                
                        'probs': np.concatenate(y_proba_posurines),                    
                    }
                )
            except:
                posurines_df = pd.DataFrame(
                    {
                        'names': np.concatenate([urinespositives_names for _ in range(len(y_preds_posurines))]),
                        'cv': np.concatenate([np.ones(len(x)) * i for i, x in enumerate(y_preds_posurines)]),
                        'batches': np.concatenate([urinespositives_batches for _ in range(len(y_preds_posurines))]),
                        'preds': np.concatenate(y_preds_posurines),
                        'labels': np.concatenate([urinespositives_real_df.loc[:, 'Class'] for _ in range(len(y_preds_posurines))]),                
                    }
                )
            posurines_df.loc[:, 'preds'] = [unique_labels[l] for l in posurines_df.loc[:, 'preds'].to_numpy()]
            assert 'bact' not in posurines_df.loc[:, 'preds']
            posurines_df.to_csv(
                f'{self.log_path}/saved_models/{self.args.model_name}_posurines_individual_results.csv'
            )
            # save the mode scores too
            y_preds_posurines = np.stack(y_preds_posurines)
            from scipy import stats
            y_preds_posurines = stats.mode(y_preds_posurines, axis=0)[0].flatten()

            try:
                posurines_df = pd.DataFrame(
                    {
                        'names': urinespositives_names,
                        'batches': urinespositives_batches,
                        'preds': y_preds_posurines,
                        'labels': urinespositives_real_df.loc[:, 'Class'],                
                        'probs': y_proba_posurines,                    
                    }
                )
            except:
                posurines_df = pd.DataFrame(
                    {
                        'names': urinespositives_names,
                        'batches': urinespositives_batches,
                        'preds': y_preds_posurines,
                        'labels': urinespositives_real_df.loc[:, 'Class'],                
                    }
                )

            posurines_df.loc[:, 'preds'] = [unique_labels[l] for l in posurines_df.loc[:, 'preds'].to_numpy()]
            assert 'bact' not in posurines_df.loc[:, 'preds']
            posurines_df.to_csv(
                f'{self.log_path}/saved_models/{self.args.model_name}_posurines_mode_results.csv'
            )

            self.best_scores_train = scores_train
            self.best_scores_valid = scores_valid
            self.best_scores_test = scores_test
            self.best_mccs_train = mccs_train
            self.best_mccs_valid = mccs_valid
            self.best_mccs_test = mccs_test


            fig = get_confusion_matrix(urinespositives_classes, 
                                       y_preds_posurines, 
                                       unique_labels)
            save_confusion_matrix(fig, 
                                  f"{self.log_path}/confusion_matrices/" 
                                  f"{self.name}_{self.args.model_name}_posurines", 
                                  acc=ACC(y_preds_posurines, urinespositives_classes), 
                                  mcc=MCC(y_preds_posurines, urinespositives_classes), 
                                  group='posurines')
            fig = get_confusion_matrix(np.concatenate(train_classes), 
                                       np.concatenate(y_preds_train), 
                                       unique_labels)
            save_confusion_matrix(fig, 
                                  f"{self.log_path}/confusion_matrices/" 
                                  f"{self.name}_{self.args.model_name}_train", 
                                  acc=scores_train, mcc=mccs_train, group='train')
            fig = get_confusion_matrix(np.concatenate(valid_classes), 
                                       np.concatenate(y_preds_valid),
                                       unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/"
                                  f"{self.name}_{self.args.model_name}_valid", 
                                  acc=scores_valid, mcc=mccs_valid, group='valid')
            fig = get_confusion_matrix(np.concatenate(test_classes), 
                                       np.concatenate(y_preds_test), 
                                       unique_labels)
            save_confusion_matrix(fig, f"{self.log_path}/confusion_matrices/"
                                  f"{self.name}_{self.args.model_name}_test", 
                                  acc=scores_test, mcc=mccs_test, group='test')
            try:
                self.best_roc_train = plot_roc(y_proba_train, train_classes, unique_labels,
                             f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_train", 
                             binary=self.binary, acc=scores_train)
            except:
                pass
            try:
                self.best_roc_valid = plot_roc(y_proba_valid, valid_classes, unique_labels,
                             f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_valid", 
                             binary=self.binary, acc=scores_valid)
            except:
                pass
            try:
                self.best_roc_test = plot_roc(y_proba_test, test_classes, unique_labels,
                             f"{self.log_path}/ROC/{self.name}_{self.args.model_name}_test",
                             binary=self.binary, acc=scores_test)
            except:
                pass
            unique_batches = {
                'train': train_unique_batches_list,
                'valid': valid_unique_batches_list,
                'test': test_unique_batches_list
            }
            self.save_best_model_hparams(param_grid, other_params, scaler_name, unique_batches, metrics)

        return 1 - np.mean(mccs_valid)

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

        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}.json', "w") as read_file:
            json.dump(self.best_params_dict, read_file)
        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}_values.json', "w") as read_file:
            json.dump(self.best_params_dict_values, read_file)
        # load model


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
    parser.add_argument('--remove_blancs', type=int, default=1)
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
    batch_date = '-'.join(batch_dates)

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

    print('N features:', data['inputs']['all'].shape[1])

    uniques = {
        'labels': unique_labels,
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
            Categorical(['minmax', 'standard', 'robust', 
                         'minmax_per_batch', 'standard_per_batch', 
                         'robust_per_batch'], name="scaler"),
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
            Categorical(['minmax', 'standard', 'robust', 'minmax2',
                         'minmax_per_batch', 'standard_per_batch', 
                         'robust_per_batch'], name="scaler"),
        ]
    elif args.model_name == 'linsvc':
        cfr = sklearn.svm.LinearSVC
        space = [
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Real(1e-4, 10, 'log-uniform', name='tol'),
            Integer(1, 10000, 'uniform', name='max_iter'),
            Categorical(['l2'], name='penalty'),
            Categorical(['hinge', 'squared_hinge'], name='loss'),
            Real(1e-3, 10000, 'uniform', name='C'),
            Categorical(['balanced'], name='class_weight'),
            Categorical(['minmax2', 'standard', 'robust'], name="scaler")
        ]
    elif args.model_name == 'knn':
        cfr = sklearn.neighbors.KNeighborsClassifier
        space = [
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(0, 10, name='n_neighbors'),
            Categorical(['uniform', 'balanced'], name='weights'),
            Categorical(['minmax2', 'standard', 'robust'], name="scaler")
        ]
    elif args.model_name == 'svclinear':
        cfr = sklearn.svm.SVC
        space = [
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Real(1e-4, 10, 'log-uniform', name='tol'),
            Integer(1, 10000, 'uniform', name='max_iter'),
            Real(1e-3, 10000, 'uniform', name='C'),
            Categorical(['balanced'], name='class_weight'),
            Categorical(['linear'], name='kernel'),
            Categorical([True], name='probability'),
            Categorical(['minmax', 'standard', 'robust',
                         'minmax_per_batch', 'standard_per_batch',
                         'robust_per_batch'], name="scaler")
        ]
    elif args.model_name == 'LDA':
        cfr = LDA
        space = [
            Integer(0, 5, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Categorical(['minmax', 'standard', 'robust', 
                         'minmax_per_batch', 'standard_per_batch', 
                         'robust_per_batch'], name="scaler"),
        ]
    elif args.model_name == 'logreg':
        cfr = sklearn.linear_model.LogisticRegression
        space = [
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
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

    hparams_names = [x.name for x in space]
    exp_name = f'{batch_date.lower()}_2models_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_remblk{args.remove_blancs}__{cropings}_{"_".join(concs)}'
    binary_exp_name = f'{batch_date.lower()}_binary1_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_{cropings}_{"_".join(concs)}'
    path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
           f'/{args.spd}spd/thr{args.threshold}/{args.train_on}/{exp_name}/{args.model_name}/'
    binary_path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
           f'/{args.spd}spd/thr{args.threshold}/{args.train_on}/{binary_exp_name}/{args.binary_model_name}/'
    train = Train(name="inputs", model=cfr, data=data, uniques=uniques, 
                  hparams_names=hparams_names, log_path=path, 
                  binary_path=binary_path, args=args, logger=None, 
                  mlops='None')
        
    res = gp_minimize(train.train, space, n_calls=30, random_state=1)

    from skopt.plots import plot_objective, plot_histogram
    _ = plot_objective(res.optimizer_results_[0],
                    dimensions=["C", "degree", "gamma", "kernel"],
                    n_minimum_search=int(1e8))
    plt.savefig('objective_skopt_linsvc.png')