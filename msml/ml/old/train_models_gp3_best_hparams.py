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
from dataset import get_data
from train import Train

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
    parser.add_argument("--low_ram", type=int, default=0)
    parser.add_argument("--remove_bad_samples", type=int, default=1)

    args = parser.parse_args()
    if args.mz < 1:
        args.mz_rounding = len(str(args.mz).split('.')[-1]) + 1
    else:
        args.mz_rounding = 1

    if args.rt < 1:
        args.rt_rounding = len(str(args.rt).split('.')[-1]) + 1
    else:
        args.rt_rounding = 1
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
    cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    args.exp_name = f'{"-".join(batches_to_keep)}_binary{args.binary}_{args.n_features}' \
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
           f'{args.train_on}/{args.exp_name}/{args.model_name}_best_hparams/remove_bad_samples{args.remove_bad_samples}'

    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    minmax_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()
    standard_minmax_scaler = Pipeline([('standard', StandardScaler()),
                                       ('minmax', MinMaxScaler())])
    robust_minmax_scaler = Pipeline([('robust', RobustScaler()),
                                     ('minmax', MinMaxScaler())])

    if args.model_name == 'xgboost':
        import xgboost
        cfr = xgboost.XGBClassifier
        space = {
            'threshold': 0.00127501,
            'n_aug': 1,
            'p': 0.00561209,
            'g': 0.168697,
            'scaler': 'zscore',
            'n_estimators': 100,
            'max_depth': 6,
        }
    else:
        exit("Model not implemented")

    train = Train(name="inputs", model=cfr, data=data, uniques=uniques,
                  hparams_names=list(space.keys()), log_path=path, args=args, 
                  logger=None, log_neptune=True, mlops='None')
    
    res = train.train_no_split(list(space.values()))

