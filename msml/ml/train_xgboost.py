import copy
import os
import neptune
import pickle
import numpy as np
import pandas as pd
from .utils import remove_zero_cols, scale_data, get_empty_lists
from .torch_utils import augment_data
from .loggings import log_ord, log_fct, log_neptune
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC
# from scipy import stats
# from .log_shap import log_shap
import xgboost
# import pipeline from sklearn
from .utils import columns_stats_over0
from .train import Train

import sys

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'


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
            print(f"Error in lists['proba']['posurines'] += [m.predict(durines)] in add_results_to_list: {e}")
            pass

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
                'names': lists['names']['valid'][-1]
            },
            'test': {
                'inputs': data_dict['data']['test'],
                'labels': lists['classes']['test'][-1],
                'preds': lists['preds']['test'][-1],
                'proba': lists['proba']['test'][-1],
                'batches': data_dict['batches']['test'],
                'names': lists['names']['test'][-1]
            }
        }
    except Exception as e:
        print(f"Error in m.predict(dmatrices['train']): {e}")
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


class Train_xgboost(Train):
    def __init__(self, name, model, data, uniques,
                 log_path, args, logger, log_neptune, mlops='None',
                 log_metrics=None, binary_path=None):
        super(Train_xgboost, self).__init__(name, model, data, uniques,
                                            log_path, args, logger, log_neptune, mlops,
                                            log_metrics, binary_path)
        # self.hparams_names = None
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

    def get_xgboost_model(self, dmatrices, param_grid):
        eval_set = [(dmatrices['train'], 'train'), (dmatrices['valid'], 'eval')]
        early_stop = xgboost.callback.EarlyStopping(int(param_grid['early_stopping_rounds']))
        if self.args.model_name == 'xgboost' and 'cuda' in self.args.device:
            if len(self.args.device.split(':')) == 0:
                gpu_id = 0
            else:
                gpu_id = self.args.device.split(':')[1]
            m = xgboost.train({
                "tree_method": "gpu_hist",
                "gpu_id": gpu_id,
                "max_depth": param_grid['max_depth'],
                "objective": 'multi:softprob',
                "num_class": len(self.unique_labels),
                "colsample_bytree": self.args.colsample_bytree,
                "max_bin": self.args.max_bin,
                
                # "subsample": 0.1,
                # "sampling_method": "gradient_based",
            }, dmatrices['train'],
                num_boost_round=param_grid['n_estimators'],
                evals=eval_set,
                callbacks=[early_stop],
                verbose_eval=True)
        elif self.args.model_name == 'xgboost':
            # dmatrices['train'] = xgboost.DMatrix(data_dict['data']['train'], label=lists['classes']['train'][-1])
            # dmatrices['valid'] = xgboost.DMatrix(data_dict['data']['valid'], label=lists['classes']['valid'][-1])
            # dmatrices['test'] = xgboost.DMatrix(data_dict['data']['test'], label=lists['classes']['test'][-1])
            eval_set = [(dmatrices['train'], 'train'), (dmatrices['valid'], 'eval')]
            m = xgboost.train({
                "tree_method": "hist",
                "max_depth": param_grid['max_depth'],
                "early_stopping_rounds": param_grid['early_stopping_rounds'],
                "num_boost_round": param_grid['n_estimators'],
                "objective": 'multi:softprob',
                "num_class": len(self.unique_labels),
            }, dmatrices['train'],
                num_boost_round=param_grid['n_estimators'],
                evals=eval_set,
                callbacks=[early_stop],
                verbose_eval=True)
        else:
            exit('Wrong model, only xgboost can be used')

        return m

    def train(self, h_params):
        # TODO upos not part of data_dict?
        # TODO preds not part of another dict?
        upos = {}
        posurines_df = None
        metrics = {}
        self.iter += 1
        # self.hparams_names = [x for x in h_params.keys()]
        hparams, param_grid, features_cutoff, threshold, scaler_name, n_aug, p, g = make_params_grid(h_params)
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

        if self.args.binary:
            all_data['labels']['all'] = np.array(
                ['blanc' if label == 'blanc' else 'bact' for label in all_data['labels']['all']]
            )
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

        # Initialize run as None
        run = None

        # Add dimension reduction visualization after filtering
        # if self.args.log_plots:
        #     log_ord(all_data, self.uniques, ord_path, scaler_name, 'inputs', run)

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
            model['log1p'] = run['log1p'] = self.args.log1p
            model['batches'] = run['batches'] = '-'.join(self.uniques['batches'])
            model['context'] = run['context'] = 'train'
            model['remove_bad_samples'] = run['remove_bad_samples'] = self.args.remove_bad_samples
            model['colsample_bytree'] = run['colsample_bytree'] = self.args.colsample_bytree
            model['sparse_matrix'] = run['sparse_matrix'] = self.args.sparse_matrix
            model['max_bin'] = run['max_bin'] = self.args.max_bin
            model['device'] = run['device'] = self.args.device

        else:
            model = None
            run = None

        all_data, scaler = scale_data(scaler_name, all_data)

        all_data = self.upos_operations(all_data)
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

        # Save columns stats
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
        best_iteration = []
        h = 0

        if self.args.groupkfold:
            self.args.n_repeats = len(np.unique(all_data['batches']['all']))
        while h < self.args.n_repeats:
            print(f'Fold: {h}')
            upos['inputs'] = all_data['inputs']['urinespositives']
            upos['labels'] = all_data['labels']['urinespositives']
            upos['batches'] = all_data['batches']['urinespositives']
            upos['names'] = all_data['names']['urinespositives']
            # concs_upos = all_data['concs']['urinespositives']
            data_dict = self.split_data(all_data, upos, h)
            print(f"Batches. Train: {np.unique(data_dict['batches_labels']['train'])},"
                  f"Valid: {np.unique(data_dict['batches_labels']['valid'])},"
                  f"Test: {np.unique(data_dict['batches_labels']['test'])}")
            data_dict = augment_dataset(data_dict, n_aug, p, g)
            lists = add_infos_to_dict(data_dict, lists, self.unique_labels)

            # TODO It should be possible to only input data_dict
            dmatrices = get_dmatrices(self.args, data_dict, lists)
            m = self.get_xgboost_model(dmatrices, param_grid)

            self.dump_model(h, m, scaler_name, lists)
            best_iteration += [m.best_iteration]
            add_results_to_list(m, upos, all_data, lists, dmatrices, data_dict, self.unique_labels)
            if self.best_scores['acc']['valid'] is None:
                self.best_scores['acc']['valid'] = 0

            h += 1
            try:
                np.concatenate(lists['proba']['posurines']).max(1)
            except Exception as e:
                print(f"Error in save_roc_curves: {e}")
                pass

        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords/"
        os.makedirs(ord_path, exist_ok=True)
        if self.args.log_plots:
            log_ord(self.data, self.uniques, ord_path, scaler_name, 'inputs', run)
        data = copy.deepcopy(self.data)
        metrics = log_fct(data, scaler_name, metrics)
        if self.args.log_plots:
            log_ord(data, self.uniques2, ord_path, f'{scaler_name}_blancs', 'inputs', run)

        try:
            np.concatenate(lists['proba']['posurines']).max(1)
        except Exception as e:
            print(f"Error in save_roc_curves: {e}")
            pass
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

        # Log in neptune the optimal iteration
        if self.log_neptune:
            model["best_iteration"] = run["best_iteration"] = np.round(np.mean([x for x in best_iteration]))
            model["model_size"] = run["model_size"] = get_size_in_mb(m)

        try:
            np.concatenate(lists['proba']['posurines']).max(1)
        except Exception as e:
            print(f"Error in save_confusion_matrices: {e}")
            print('DIM PROBLEM')
        lists, posurines_df = self.save_confusion_matrices(lists, run)
        self.save_calibration_curves(lists, run)
        self.save_roc_curves(lists, run)
        if np.mean(lists['mcc']['valid']) > np.mean(self.best_scores['mcc']['valid']):
            try:
                self.log_xgboost_features_ord(m, all_data, scaler_name, run)
            except:
                pass
            self.log_shap(m, lists, data_dict, run)
            # save the features kept
            with open(f'{self.log_path}/saved_models/columns_after_threshold_{scaler_name}_tmp.pkl', 'wb') as f:
                pickle.dump(data_dict['data']['test'].columns, f)

            self.keep_models(scaler_name)
            # Save the individual scores of each sample with class, #batch
            self.save_results_df(lists, run)
            self.retrieve_best_scores(lists)
            best_scores = self.save_best_model_hparams(param_grid, other_params, scaler_name,
                                                       lists['unique_batches'], metrics)
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

        if self.log_neptune:
            log_neptune(run, lists, best_scores)
            run.stop()
            model.stop()

        return 1 - np.mean(lists['mcc']['valid'])

    def log_xgboost_features_ord(self, m, all_data, scaler_name, run):
        # log ords using only the used features
        ord_path = f"{'/'.join(self.log_path.split('/')[:-1])}/ords_filtered/"
        os.makedirs(ord_path, exist_ok=True)
        if self.args.log_plots:
            # Get features used by XGBoost model
            feature_importance = m.get_score(importance_type='weight')
            # Only keep features with non-zero importance
            used_features = [f for f, imp in feature_importance.items() if imp > 0]

            # Save XGBoost feature importance
            xgb_importance_df = pd.DataFrame({
                'feature': list(feature_importance.keys()),
                'importance': list(feature_importance.values())
            }).sort_values('importance', ascending=False)
            xgb_importance_df.to_csv(f'{ord_path}/xgboost_feature_importance_{scaler_name}.csv', index=False)
            if run is not None:
                run[f'feature_importance/xgboost_{scaler_name}'].upload(
                    f'{ord_path}/xgboost_feature_importance_{scaler_name}.csv'
                )

            # Filter data_dict to only include used features
            filtered_data_dict = copy.deepcopy(all_data)
            filtered_data_dict['inputs']['all'] = filtered_data_dict['inputs']['all'][used_features]
            filtered_data_dict['inputs']['urinespositives'] =\
                filtered_data_dict['inputs']['urinespositives'][used_features]

            # Log ord with filtered features
            log_ord(filtered_data_dict, self.uniques, ord_path, scaler_name, 'filtered', run)
            log_ord(filtered_data_dict, self.uniques2, ord_path, scaler_name, 'filtered_blancs', run)

            # Get SHAP values for feature selection
            import shap
            explainer = shap.TreeExplainer(m)
            shap_values = explainer.shap_values(all_data['inputs']['all'])

            # If multiclass, take mean absolute SHAP values across classes
            if isinstance(shap_values, list):
                mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values)

            # Get mean absolute SHAP value for each feature
            mean_shap_per_feature = np.mean(mean_shap, axis=0)
            feature_names = all_data['inputs']['all'].columns

            # Save SHAP feature importance
            shap_importance_df = pd.DataFrame({
                'feature': feature_names,
                'shap_importance': mean_shap_per_feature
            }).sort_values('shap_importance', ascending=False)
            shap_importance_df.to_csv(f'{ord_path}/shap_feature_importance_{scaler_name}.csv', index=False)
            if run is not None:
                run[f'feature_importance/shap_{scaler_name}'].upload(
                    f'{ord_path}/shap_feature_importance_{scaler_name}.csv'
                )

            # Select features with non-zero SHAP values
            shap_used_features = [f for f, sv in zip(feature_names, mean_shap_per_feature) if sv.sum() > 0]

            # Create new filtered data dict with SHAP-selected features
            shap_filtered_data_dict = copy.deepcopy(all_data)
            shap_filtered_data_dict['inputs']['all'] = shap_filtered_data_dict['inputs']['all'][shap_used_features]
            shap_filtered_data_dict['inputs']['urinespositives'] =\
                shap_filtered_data_dict['inputs']['urinespositives'][shap_used_features]

            # Log ord with SHAP-filtered features
            log_ord(shap_filtered_data_dict, self.uniques, ord_path, scaler_name, 'shap', run)
            log_ord(shap_filtered_data_dict, self.uniques2, ord_path, scaler_name, 'shap_blancs', run)
