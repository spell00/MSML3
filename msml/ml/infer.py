# NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
# NEPTUNE_PROJECT_NAME = "Bacteria-MSMS"
# NEPTUNE_MODEL_NAME = 'BAC-'
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZjRiZGUzYS1kNTJmLTRkNGItOWU1MS1iNDU3MGE1NjAyODAifQ=="
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'
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
# import mattews correlation from sklearn
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
from scipy import stats
from log_shap import log_shap
import xgboost
import matplotlib.pyplot as plt
import joblib

class Infer:
    def __init__(self, name, model, data, uniques,
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
        self.best_scores = {
            'acc': {
                'test': -1,
                'posurines': -1,
            },
            'mcc': {
                'test': -1,
                'posurines': -1,
            }
        }
        if binary_path:
            self.best_scores['bact_acc'] = {
                'test': -1,
                'posurines': -1,
            }
            self.best_scores['bact_mcc'] = {
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

    def infer(self):
        metrics = {}
        self.iter += 1
        features_cutoff = None
        param_grid = {}
        scaler_name = self.args.scaler_name
        hparams = {}

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
        all_data, scaler = scale_data(scaler_name, all_data)

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
            if 'inference' in self.args.model_name:
                name = self.args.model_name.split('_')[0]
            else:
                name = self.args.model_name
            model = neptune.init_model_version(
                model=f'{NEPTUNE_MODEL_NAME}{name}{self.args.binary}',
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

        else:
            model = None
            run = None

        # import array of columns from columns_after_threshold.pkl
        with open(f'{self.log_path}/columns_after_threshold.pkl', 'rb') as f:
            columns = pickle.load(f)

        all_data['inputs']['all'] = all_data['inputs']['all'][columns]
        all_data['inputs']['urinespositives'] = all_data['inputs']['urinespositives'][columns]
        all_data['inputs']['test'] = all_data['inputs']['test'][columns]

        all_data, scaler = scale_data(scaler_name, all_data)

        print(f'Iteration: {self.iter}')
        # models = []
        # h = 0
        # seed = 0

        lists['names']['posurines'] += [np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])]
        lists['batches']['posurines'] += [all_data['batches']['urinespositives']]

        lists['names']['test'] += [all_data['names']['all']]

        lists['batches']['test'] += [all_data['batches']['all']]
        lists['unique_batches']['test'] += [list(np.unique(all_data['batches']['all']))]

        test_data = all_data['inputs']['all'].fillna(0)

        lists['classes']['test'] += [np.array([np.argwhere(l == self.unique_labels)[0][0] for l in all_data['labels']['all']])]
        lists['labels']['test'] += [all_data['labels']['all']]

        # Load all pkl files
        # models = [joblib.load(f'{self.args.exp_name}/{self.args.model_name}_minmax2_{i}.pkl') for i in [0, 1, 2, 3, 4]]
        # models += [m]

        try:
            lists['proba']['test'] += [[self.model[i].predict_proba(test_data) for i in range(len(self.model))]]
            # take the average of the proba
            lists['proba']['test'][-1] = np.mean(np.stack(lists['proba']['test'][-1]), 0)
            # Take the highest proba to make a pred
            lists['preds']['test'] += [np.argmax(lists['proba']['test'][-1], axis=1)]

        except:
            lists['preds']['test'] += [[self.model[i].predict(test_data.values) for i in range(len(self.model))]]
            # take the mode of the predictions
            lists['preds']['test'] = [stats.mode(np.stack(lists['preds']['test']), axis=0)[0].flatten()]

        if all_data['inputs']['urinespositives'].shape[0] > 0:
            try:
                lists['preds']['posurines'] += [m.predict(all_data['inputs']['urinespositives'])]
                try:
                    lists['proba']['posurines'] += [m.predict_proba(all_data['inputs']['urinespositives'])]
                except:
                    pass
                lists['mcc']['posurines'] += [MCC(lists['classes']['posurines'][-1], lists['preds']['posurines'][-1])]
                lists['acc']['posurines'] += [ACC(lists['classes']['posurines'][-1], lists['preds']['posurines'][-1])]

            except:
                lists['preds']['posurines'] += [m.predict(all_data['inputs']['urinespositives'].values)]
                try:
                    lists['proba']['posurines'] += [m.predict_proba(all_data['inputs']['urinespositives'].values)]
                except:
                    pass
                lists['mcc']['posurines'] += [MCC(lists['classes']['posurines'][-1], lists['preds']['posurines'][-1])]
                lists['acc']['posurines'] += [ACC(lists['classes']['posurines'][-1], lists['preds']['posurines'][-1])]
            
        try:
            data_list = {
                'test': {
                    'inputs': test_data,
                    'labels': lists['classes']['test'][-1],
                    'preds': lists['preds']['test'][-1],
                    'proba': lists['proba']['test'][-1],
                    'batches': all_data['batches']['all'],
                    'names': all_data['names']['test']
                },
            }
        except:
            data_list = {
                'test': {
                    'inputs': test_data,
                    'labels': lists['classes']['test'][-1],
                    'preds': lists['preds']['test'][-1],
                    'batches': all_data['batches']['all'],
                    'names': all_data['names']['test']
                },
            }
        lists['mcc']['test'] += [MCC(lists['classes']['test'][-1], lists['preds']['test'][-1])]
        lists['acc']['test'] += [ACC(lists['classes']['test'][-1], lists['preds']['test'][-1])]
        
        # TODO all_data and self.data and data are all messed up
        self.data['inputs']['all'] = all_data['inputs']['all']
        self.data['inputs']['urinespositives'] = all_data['inputs']['urinespositives']
        self.data['inputs']['test'] = all_data['inputs']['test']

        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        log_ord(self.data, self.uniques, ord_path, scaler_name, run)
        data = copy.deepcopy(self.data)
        metrics = log_fct(data, scaler_name, metrics)
        log_ord(data, self.uniques2, ord_path, f'{scaler_name}_blancs', run)

        print(lists['mcc']['valid'])
        print('test_acc:', np.mean(lists['acc']['test']), \
              'test_mcc:', np.mean(lists['mcc']['test']), \
              'scaler:', scaler_name,
              'h_params:', param_grid
              )
        lists = self.save_confusion_matrices(lists, run)
        self.save_roc_curves(lists, run)
        # if np.mean(lists['mcc']['test']) > np.mean(self.best_scores['mcc']['test']):
            # log_shap(run, m, data_list, all_data['inputs']['all'].columns, self.bins, self.log_path)
            # Save the individual scores of each sample with class, #batch
        self.save_results_df(lists, run)
            # best_scores = self.save_best_model_hparams(param_grid, other_params, scaler_name, lists['unique_batches'], metrics)
        # else:
        #     best_scores = {
        #         'nbe': None,
        #         'ari': None,
        #         'ami': None,
        #     }

        if self.log_neptune:
            log_neptune(run, lists, None)
            run.stop()
            model.stop()

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

        self.best_params_dict_values['test_acc'] = self.best_scores['acc']['test']
        self.best_params_dict_values['test_acc'] = self.best_scores['acc']['test']
        self.best_params_dict_values['test_mcc'] = self.best_scores['mcc']['test']
        self.best_params_dict_values['test_mcc'] = self.best_scores['mcc']['test']
        
        self.best_params_dict['ami'] = metrics[scaler_name]['all']['adjusted_mutual_info_score']['domains']
        self.best_params_dict['ari'] = metrics[scaler_name]['all']['adjusted_rand_score']['domains']
        n_batches = np.unique(
            np.concatenate([
                np.concatenate(unique_batches[b]) for b in unique_batches if len(unique_batches[b]) > 0
            ])
        ).flatten().shape[0]
        self.best_params_dict['nBE'] = (np.log(n_batches) - metrics[scaler_name]['all']['shannon']['domains'])  / np.log(n_batches)

        self.best_params_dict['test_acc_mean'] = np.mean(self.best_scores['acc']['test'])
        self.best_params_dict['test_acc_std'] = np.std(self.best_scores['acc']['test'])

        self.best_params_dict['test_mcc_mean'] = np.mean(self.best_scores['mcc']['test'])
        self.best_params_dict['test_mcc_std'] = np.std(self.best_scores['mcc']['test'])

        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}.json', "w") as read_file:
            json.dump(self.best_params_dict, read_file)
        with open(f'{self.log_path}/saved_models/best_params_{self.name}_{self.args.model_name}_values.json', "w") as read_file:
            json.dump(self.best_params_dict_values, read_file)
        # add ari, ami and nbe to run
        return self.best_params_dict

    def make_predictions(self, all_data, lists, run):
        urinespositives_names = np.array([x.split('_')[-2] for x in all_data['names']['urinespositives']])
        urinespositives_batches = np.array([x for x in all_data['batches']['urinespositives']])
        # TODO load expected classes elsewhere
        urinespositives_real_df = pd.concat((
            pd.read_csv(f'resources/bacteries_2024/B10-05-03-2024/b10_patients_samples.csv'),
            pd.read_csv(f'resources/bacteries_2024/B11-05-24-2024/b11_patients_samples.csv')
        ))
        # remove names that are not in the real df
        urinespositives_names = np.array([x for x in urinespositives_names if x in urinespositives_real_df.loc[:, 'ID'].to_numpy()])

        urinespositives_real_df.loc[:, 'Class'] = [l.lower() for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]
        if self.args.binary:
            urinespositives_real_df.loc[:, 'Class'] = ['blanc' if l == 'blanc' else 'bact' for l in urinespositives_real_df.loc[:, 'Class'].to_numpy()]

        to_keep = np.argwhere(np.isin(urinespositives_names, urinespositives_real_df.loc[:, 'ID'].to_numpy()) == True).flatten()
        # to_keep2 = np.argwhere(np.isin(urinespositives_real_df.loc[:, 'ID'].to_numpy(), urinespositives_names) == True).flatten()
        # take intersect
        # to_keep = np.intersect1d(to_keep, to_keep2)
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
        if len(lists['proba']['test']) > 0:
            df_test = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['test']),
                    'labels': np.concatenate(lists['labels']['test']),
                    'batches': np.concatenate(lists['batches']['test']),
                    'preds': np.concatenate(lists['preds']['test']),
                    'proba': np.concatenate(lists['proba']['test']).max(1),
                    'names': np.concatenate(lists['names']['test'])
                }
            )
            for i, label in enumerate(self.unique_labels):
                df_test[label] = np.concatenate(lists['proba']['test'])[:, i]
                
        else:
            df_test = pd.DataFrame(
                {
                    'classes': np.concatenate(lists['classes']['test']),
                    'labels': np.concatenate(lists['labels']['test']),
                    'batches': np.concatenate(lists['batches']['test']),
                    'preds': np.concatenate(lists['preds']['test']),
                    'names': np.concatenate(lists['names']['test'])
                }
            )

        df_test.loc[:, 'preds'] = [
            self.unique_labels[l] for l in df_test.loc[:, 'preds'].to_numpy()
        ]

        df_test.to_csv(f'{self.log_path}/saved_models/{self.args.model_name}_test_individual_results.csv')
        run[f'test/individual_results'].upload(f'{self.log_path}/saved_models/{self.args.model_name}_test_individual_results.csv')
        self.save_thresholds_curve('test', lists, run)
        self.save_thresholds_curve0('test', df_test, run)
        # plot_bars(self.args, run, self.unique_labels)
        

    def save_roc_curves(self, lists, run):
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
            df = pd.DataFrame(
                {
                    'classes': lists['classes'][group][batch],
                    'labels': lists['labels'][group][batch],
                    'batches': lists['batches'][group][batch],
                    'preds': lists['preds'][group][batch], 
                    'proba': lists['proba'][group][batch].max(1),
                    'names': lists['names'][group][batch]
                }
            )
            for i, label in enumerate(self.unique_labels):
                df[label] = lists['proba'][group][batch][:, i]
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

        
    def save_confusion_matrices(self, lists, run):
        relevant_samples = []
        # posurines_df = self.make_predictions(all_data, lists, run)
        # relevant_samples = [i for i, l in enumerate(posurines_df.loc[:, 'labels'].to_numpy()) if l in self.unique_labels]
        # if len(relevant_samples) > 0:
        #     posurines_df = posurines_df.iloc[relevant_samples]
        #     posurines_classes = [int(np.argwhere(l == self.unique_labels).flatten()) for l in posurines_df.loc[:, 'labels'].to_numpy()]
        #     posurines_preds = [int(np.argwhere(l == self.unique_labels).flatten()) for l in posurines_df.loc[:, 'preds'].to_numpy()]
        #     lists[f'acc']['posurines'] = [ACC(posurines_preds, posurines_classes)]
        #     lists[f'mcc']['posurines'] = [MCC(posurines_preds, posurines_classes)]
        #     lists['classes']['posurines'] = [posurines_classes]
        #     lists['preds']['posurines'] = [posurines_preds]
        if len(relevant_samples) > 0:
            groups = ['test', 'posurines']
        else:
            groups = ['test']
        for group in groups:
            fig = get_confusion_matrix(np.concatenate(lists['classes'][group]), 
                                        np.concatenate(lists['preds'][group]), 
                                        self.unique_labels)
            save_confusion_matrix(fig, 
                                    f"{self.log_path}/confusion_matrices/" 
                                    f"{self.name}_{self.args.model_name}_{group}", 
                                    acc=lists['acc'][group], mcc=lists['mcc'][group], group=group)
            run[f'confusion_matrix/{group}'].upload(fig)


        return lists



