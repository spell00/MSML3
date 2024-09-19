import os
# NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
# NEPTUNE_PROJECT_NAME = "Bacteria-MSMS"
# NEPTUNE_MODEL_NAME = 'BAC-'
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZjRiZGUzYS1kNTJmLTRkNGItOWU1MS1iNDU3MGE1NjAyODAifQ=="
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'
import numpy as np
import random
import sklearn.neighbors
# import torch
import sklearn
import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import warnings
# TODO scikit-optimize is discontinued, need to change this. Only works with python 3.8
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

random.seed(42)
# torch.manual_seed(42)
np.random.seed(42)

from skopt import gp_minimize
import matplotlib.pyplot as plt
from dataset import get_data_all
from train import Train
import cupy

if __name__ == '__main__':
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--features_file', type=str, default='mutual_info_classif_scores.csv')
    parser.add_argument('--remove_zeros', type=int, default=0)
    parser.add_argument('--log1p', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--pool', type=int, default=0)
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='linsvc')
    parser.add_argument('--train_on', type=str, default='all')
    parser.add_argument('--csv_file', type=str, default='inputs')
    parser.add_argument('--ovr', type=int, default=1)
    parser.add_argument('--mz', type=float, default=10)
    parser.add_argument('--rt', type=float, default=10)
    parser.add_argument('--mzp', type=float, default=10)
    parser.add_argument('--rtp', type=float, default=10)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--spd', type=int, default=200)
    parser.add_argument('--ms_level', type=int, default=2)
    parser.add_argument('--combat', type=int, default=0)  # TODO not using this anymore
    parser.add_argument('--shift', type=int, default=0)  # TODO keep this?
    parser.add_argument('--log', type=str, default='inloop')
    parser.add_argument('--features_selection', type=str, default='none')
    parser.add_argument('--concs', type=str, default='na,h')
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument("--min_mz", type=int, default=0)
    parser.add_argument("--max_mz", type=int, default=10000)
    parser.add_argument("--min_mz_parent", type=int, default=359)
    parser.add_argument("--max_mz_parent", type=int, default=872)
    parser.add_argument("--min_rt", type=int, default=0)
    parser.add_argument("--max_rt", type=int, default=1000)
    parser.add_argument("--low_ram", type=int, default=0)
    parser.add_argument("--remove_bad_samples", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--log_shap", type=int, default=1)
    parser.add_argument("--fp", type=str, default=32)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--max_bin", type=int, default=256)
    parser.add_argument("--sparse_matrix", type=int, default=0)

    args = parser.parse_args()
    if args.mz < 1:
        args.mz_rounding = len(str(args.mz).split('.')[-1]) + 1
    else:
        args.mz_rounding = 1
        args.mz = int(args.mz)
        args.mzp = int(args.mzp)

    if args.rt < 1:
        args.rt_rounding = len(str(args.rt).split('.')[-1]) + 1
    else:
        args.rt_rounding = 1
        args.rt = int(args.rt)
        args.rtp = int(args.rtp)
    args.csv_file = f'{args.csv_file}_{args.features_selection}.csv'
    args.features_file = f'{args.features_selection}_scores.csv'
    args.bins = {
        'mz_bin': args.mz,
        'rt_bin': args.rt,
        'mz_max': args.max_mz,
        'rt_max': args.max_rt,
        'mz_rounding': args.mz_rounding,
        'rt_rounding': args.rt_rounding
    }

    batch_dates = [
        # "B15-06-29-2024",
        "B14-06-10-2024", "B13-06-05-2024", "B12-05-31-2024", "B11-05-24-2024",
        "B10-05-03-2024", "B9-04-22-2024", "B8-04-15-2024",
        'B7-04-03-2024', 'B6-03-29-2024', 'B5-03-13-2024',
        'B4-03-01-2024', 'B3-02-29-2024', 'B2-02-21-2024',
        'B1-02-02-2024'
    ]
    batches_to_keep = [
        # "b15-06-29-2024",
        "b14-06-10-2024", "b13-06-05-2024", "b12-05-31-2024", "b11-05-24-2024",
        "b10-05-03-2024", "b9-04-22-2024", "b8-04-15-2024", 
        'b7-04-03-2024', 'b6-03-29-2024', 'b5-03-13-2024', 
        'b4-03-01-2024', 'b3-02-29-2024', 'b2-02-21-2024', 
        'b1-02-02-2024'
    ]
    batch_dates = [x.split('-')[0] for x in batch_dates]
    batches_to_keep = [x.split('-')[0] for x in batches_to_keep]

    concs = args.concs.split(',')
    cropings = f"mz0-10000rt0-320"
    new_cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    args.exp_name = f'{"-".join(batches_to_keep)}_binary{args.binary}_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_{new_cropings}_{"_".join(concs)}'

    # TODO change gkf0; only valid because using all features
    exp = f'all_{"-".join(batch_dates)}_gkf0_{cropings}_5splits'  
    path = f'resources/bacteries_2024/matrices/mz{args.mz}/rt{args.rt}/' \
        f'mzp{args.mzp}/rtp{args.rtp}/thr{args.threshold}/{args.spd}spd/' \
        f'ms{args.ms_level}/combat{args.combat}/shift{args.shift}/none/' \
        f'log{args.log}/{args.features_selection}/{exp}'

    data, unique_labels, unique_batches, unique_manips, \
                        unique_urines, unique_concs = get_data_all(path, args)
    
    if args.fp == 'float16':
        data['inputs']['all'] = data['inputs']['all'].astype(np.float16)
    elif args.fp == 'float32':
        data['inputs']['all'] = data['inputs']['all'].astype(np.float32)
    elif args.fp == 'float64':
        data['inputs']['all'] = data['inputs']['all'].astype(np.float64)
        

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

    # TODO Assert that no concentration is left that should not
    try:
        assert len(np.unique(data['concs']['all'])) == len(concs)
    except:
        raise ValueError(f'Problem with concentrations: Should have: {concs} but got {np.unique(data["concs"]["all"])}')
    path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/{args.spd}spd/thr{args.threshold}/' \
           f'{args.train_on}/{args.exp_name}/{args.model_name}/'

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
    elif args.model_name == 'xgboost':
        print('XGBOOST')
        import xgboost
        from train_xgboost import Train
        cfr = xgboost.XGBClassifier
        space = [
            Real(0, 0.5, 'uniform', name='threshold'),
            # Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(4, 5, 'uniform', name='max_depth'), # BEST WAS DEFAULT
            Real(10, 20, 'uniform', name='early_stopping_rounds'), # 
            Integer(100, 150, 'uniform', name='n_estimators'),
            # Categorical(['binary:logistic'], name='objective'),
            Categorical(['minmax2'], name="scaler"),
        ]
    elif args.model_name == 'xgboostda':
        print('XGBOOST DASK')
        import xgboost
        from train_xgboost_dask import Train
        cfr = xgboost.XGBClassifier
        space = [
            Real(0, 0.5, 'uniform', name='threshold'),
            # Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(4, 5, 'uniform', name='max_depth'), # BEST WAS DEFAULT
            Real(10, 20, 'uniform', name='early_stopping_rounds'), # 
            Integer(100, 150, 'uniform', name='n_estimators'),
            # Categorical(['binary:logistic'], name='objective'),
            Categorical(['minmax2'], name="scaler"),
        ]
    elif args.model_name == 'xgboostex':
        import xgboost
        from train_xgboost_extmem import Train
        cfr = xgboost.XGBClassifier
        space = [
            Real(0, 0.5, 'uniform', name='threshold'),
            Integer(0, 1, 'uniform', name='n_aug'),
            Real(0, 0.5, 'uniform', name='p'),
            Real(0, 0.5, 'uniform', name='g'),
            Integer(5, 9, 'uniform', name='max_depth'), # BEST WAS DEFAULT
            Real(10, 20, 'uniform', name='early_stopping_rounds'), # 
            Integer(300, 350, 'uniform', name='n_estimators'),
            # Categorical(['binary:logistic'], name='objective'),
            Categorical(['minmax2'], name="scaler"),
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
