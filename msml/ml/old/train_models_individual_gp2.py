import os
import copy
import json
import pandas as pd
import numpy as np
import random
import pickle
import warnings
import matplotlib.pyplot as plt

import torch

import sklearn
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

from utils import scale_data
from loggings import log_ord, log_fct, save_confusion_matrix
from utils import augment_data         
from dataset import get_data
from sklearn_train_nocv import count_labels, get_confusion_matrix, save_roc_curve, plot_roc

warnings.filterwarnings("ignore")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

from train import Train
class iTrain(Train):
    def __init__(self, name, model, data, uniques, hparams_names, log_path, 
                 args, logger, log_neptune, mlops='None', binary_path=None):
        super().__init__(name, model, data, uniques, hparams_names, log_path, 
                         args, logger, log_neptune, mlops, binary_path)
        self.model_name = f'indiv_binary{self.args.binary}_{self.args.model_name}'


if __name__ == '__main__':
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, default='mutual_info_classif_scores.csv')
    parser.add_argument('--remove_zeros', type=int, default=0)
    parser.add_argument('--log1p', type=int, default=0)
    parser.add_argument('--pool', type=int, default=0)
    parser.add_argument('--n_features', type=int, default=100)
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
    parser.add_argument('--classif_loss', type=str, default='celoss')
    parser.add_argument("--min_mz", type=int, default=0)
    parser.add_argument("--max_mz", type=int, default=10000)
    parser.add_argument("--min_rt", type=int, default=0)
    parser.add_argument("--max_rt", type=int, default=1000)
    args = parser.parse_args()
    args.groupkfold = 0
    batch_dates = [
        'B10-05-03-2024',
        'B1-02-02-2024', 'B2-02-21-2024', 'B3-02-29-2024', 
        'B4-03-01-2024', 'B5-03-13-2024', 'B6-03-29-2024',
        'B7-04-03-2024', "B8-04-15-2024", "B9-04-22-2024",
    ]
    batch_dates = [x.split('-')[0] for x in batch_dates]
    batch_date = '-'.join(batch_dates)
    cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    for batches_to_keep in batch_dates:
        # if batches_to_keep in ['B1', 'B2', 'B3', 'B4', 'B6', 'B7', 'B8', 'B9']:
        #     continue
        print('BATCH:', batches_to_keep)
        concs = args.concs.split(',')
        batches_to_keep = [batches_to_keep]
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

        batches_to_keep = [x.lower() for x in batches_to_keep]
        mask = [True if x in batches_to_keep else False for x in data['batches']['all']]
        data['inputs']['all'] = data['inputs']['all'].iloc[mask]
        data['inputs']['all'].iloc[:] = np.nan_to_num(data['inputs']['all'])
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
        
        unique_labels = np.array(np.unique(data['labels']['all']))
        unique_batches = np.array(np.unique(data['batches']['all']))
        unique_manips = np.array(np.unique(data['manips']['all']))
        unique_urines = np.array(np.unique(data['urines']['all']))
        unique_concs = np.array(np.unique(data['concs']['all']))
        
        uniques = {
            'labels': unique_labels,
            'batches': unique_batches,
            'manips': unique_manips,
            'urines': unique_urines,
            'concs': unique_concs
        }

        # TODO REMOVE MULTI
        # path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
        #     f'{args.train_on}/{exp_name}/{args.model_name}/'
        path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/{args.spd}spd/thr{args.threshold}/' \
            f'{args.train_on}/{exp_name}/{args.model_name}/'
        os.makedirs(path, exist_ok=True)
        if args.model_name == 'RF':
            cfr = RandomForestClassifier
            space = [
                # Integer(100, 20000, 'uniform', name='features_cutoff'),
                # Real(0, 1, 'uniform', name='threshold'),
                Real(0, 1, 'uniform', name='threshold'),
                Integer(0, 1, 'uniform', name='n_aug'),
                Real(0, 0.5, 'uniform', name='p'),
                Real(0, 0.5, 'uniform', name='g'),
                Integer(1, 100, 'uniform', name="max_features"),
                Integer(2, 10, 'uniform', name="min_samples_split"),
                Integer(1, 10, 'uniform', name="min_samples_leaf"),
                Integer(1, 1000, 'uniform', name="n_estimators"),
                Categorical(['gini', 'entropy'], name="criterion"),
                Categorical([True, False], name="oob_score"),
                Categorical(['balanced'], name="class_weight"),
                Categorical(['minmax2', 'robust', 'minmax', 'standard'], name="scaler")
            ]
        elif args.model_name == 'linsvc':
            cfr = sklearn.svm.LinearSVC
            space = [
                Real(0, 1, 'uniform', name='threshold'),
                Integer(0, 1, 'uniform', name='n_aug'),
                Real(0, 0.5, 'uniform', name='p'),
                Real(0, 0.5, 'uniform', name='g'),
                Real(1e-4, 10, 'log-uniform', name='tol'),
                Integer(1, 10000, 'uniform', name='max_iter'),
                Categorical(['l2'], name='penalty'),
                Categorical(['hinge', 'squared_hinge'], name='loss'),
                Real(1e-3, 10000, 'uniform', name='C'),
                Categorical(['balanced'], name='class_weight'),
                Categorical(['minmax2', 'robust', 'minmax', 'standard'], name="scaler")
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
                Categorical(['minmax2', 'robust', 'minmax', 'standard'], name="scaler")
            ]
        elif args.model_name == 'LDA':
            cfr = LDA
            space = [
                Real(0, 1, 'uniform', name='threshold'),
                Integer(0, 5, 'uniform', name='n_aug'),
                Real(0, 0.5, 'uniform', name='p'),
                Real(0, 0.5, 'uniform', name='g'),
                Categorical(['minmax2', 'robust', 'minmax', 'standard'], name="scaler")
            ]
        elif args.model_name == 'logreg':
            cfr = sklearn.linear_model.LogisticRegression
            space = [
                Real(0, 1, 'uniform', name='threshold'),
                Integer(0, 1, 'uniform', name='n_aug'),
                Real(0, 0.5, 'uniform', name='p'),
                Real(0, 0.5, 'uniform', name='g'),
                Integer(1, 20000, 'uniform', name='max_iter'),
                Real(1e-3, 20000, 'uniform', name='C'),
                Categorical(['saga'], name='solver'),
                Categorical(['l1', 'l2'], name='penalty'),
                Categorical([True, False], name='fit_intercept'),
                Categorical(['balanced'], name='class_weight'),
                Categorical(['minmax2', 'robust', 'minmax', 'standard'], name="scaler")
            ]
        elif args.model_name == 'NB':
            cfr = sklearn.naive_bayes.GaussianNB
            space = [
                Real(0, 1, 'uniform', name='threshold'),
                Integer(0, 5, 'uniform', name='n_aug'),
                Real(0, 0.5, 'uniform', name='p'),
                Real(0, 0.5, 'uniform', name='g'),
                Categorical(['minmax2', 'robust', 'minmax', 'standard'], name="scaler")
            ]
                        
        hparams_names = [x.name for x in space]
        train = iTrain(name="inputs", model=cfr, data=data, uniques=uniques,
                      hparams_names=hparams_names, log_path=path, args=args,
                      logger=None, log_neptune=True, mlops='None')
        
        res = gp_minimize(train.train, space, n_calls=30, random_state=1)

