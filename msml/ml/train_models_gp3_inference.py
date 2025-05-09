import os
# NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
# NEPTUNE_PROJECT_NAME = "Bacteria-MSMS"
# NEPTUNE_MODEL_NAME = 'BAC-'
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZjRiZGUzYS1kNTJmLTRkNGItOWU1MS1iNDU3MGE1NjAyODAifQ=="
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'
import numpy as np
import random
import os
import warnings

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

import matplotlib.pyplot as plt
from dataset import get_data_infer, get_data
from infer import Infer
import time
import joblib


if __name__ == '__main__':
    # Start timer
    start = time.time()
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, default='none_scores.csv')
    parser.add_argument('--log1p', type=int, default=0)
    parser.add_argument('--zinb', type=int, default=0)
    parser.add_argument('--groupkfold', type=int, default=0)
    parser.add_argument('--pool', type=int, default=0)
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='linsvc')
    parser.add_argument('--train_on', type=str, default='all')
    parser.add_argument('--csv_file', type=str, default='inputs_none.csv')
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
    parser.add_argument('--features_selection', type=str, default='none')
    parser.add_argument('--concs', type=str, default='na,h')
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument("--min_mz", type=int, default=0)
    parser.add_argument("--max_mz", type=int, default=10000)
    parser.add_argument("--min_rt", type=int, default=0)
    parser.add_argument("--max_rt", type=int, default=1000)
    parser.add_argument("--min_mz_parent", type=int, default=359)
    parser.add_argument("--max_mz_parent", type=int, default=872)
    parser.add_argument("--low_ram", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--train_batches", type=str, default='b14-b13-b12-b11-b10-b9-b8-b7-b6-b5-b4-b3-b2-b1')
    parser.add_argument("--scaler_name", type=str, default='minmax2')
    parser.add_argument("--fp", type=str, default=32)
    # parser.add_argument("--colsample_bytree", type=float, default=1.0)
    # parser.add_argument("--max_bin", type=int, default=256)
    parser.add_argument("--sparse_matrix", type=int, default=1)
    parser.add_argument("--remove_bad_samples", type=int, default=1)

    args = parser.parse_args()

    args.model_name = f"{args.model_name}"

    if args.mz < 1:
        args.mz_rounding = len(str(args.mz).split('.')[-1]) + 1
    else:
        args.mz_rounding = 1

    if args.rt < 1:
        args.rt_rounding = len(str(args.rt).split('.')[-1]) + 1
    else:
        args.rt_rounding = 1
    args.bins = {
        'mz_bin': args.mz,
        'rt_bin': args.rt,
        'mz_max': args.max_mz,
        'rt_max': args.max_rt,
        'mz_rounding': args.mz_rounding,
        'rt_rounding': args.rt_rounding
    }
    args.remove_zeros = 0  # Has to be 0 for inference

    batch_dates = [
        "B15-06-29-2024"
    ]
    batches_to_keep = [
        "b15-06-29-2024"
    ]
    batch_dates = [x.split('-')[0] for x in batch_dates]
    batches_to_keep = [x.split('-')[0] for x in batches_to_keep]

    concs = args.concs.split(',')
    # cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    cropings = f"mz0-10000rt0-320"

    # TODO change gkf0; only valid because using all features    
    args.exp_name = f'results/multi/mz{args.mz}/rt{args.rt}/' \
        f'ms{args.ms_level}/{args.spd}spd/thr{args.threshold}/' \
        f'{args.train_on}/{args.train_batches}_binary{args.binary}_{args.n_features}_' \
        f'gkf{args.groupkfold}_ovr{args.ovr}_{cropings}_{"_".join(concs)}/xgboost/saved_models/'
    exp = f'all_{"-".join(batch_dates)}_gkf{args.groupkfold}_{cropings}_5splits'  
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

    models = [joblib.load(f'{args.exp_name}/{args.model_name}_{args.scaler_name}_{i}.pkl') for i in [0, 1, 2, 3, 4]]
    names = [x.split('.')[0] for x in os.listdir(path) if x.endswith('.pkl')]
    args.path = path
    # args.scaler_name = 'minmax2'  # changer pour que ce soit automatique
    infer = Infer(name="inputs", model=models, data=data, uniques=uniques,
                  log_path=path, args=args, logger=None,
                  log_neptune=True, mlops='None')
    
    infer.infer()

    # End timer
    end = time.time()
    print(f"Time taken: {end - start} seconds")