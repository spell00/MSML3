import os
import xgboost
import numpy as np
import random
import torch
import sklearn.neighbors
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import warnings
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
try:
    from .dataset import get_data_all  # when run as module: python -m msml.ml.train_models_gp3
except ImportError:
    # Fallback when executed directly: python msml/ml/train_models_gp3.py
    import os, sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(os.path.dirname(this_dir))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from msml.ml.dataset import get_data_all
    from msml.ml.train_xgboost import Train_xgboost
    from msml.ml.train_bernn import Train_bernn
from sklearn.naive_bayes import GaussianNB
from bernn import TrainAEClassifierHoldout
from msml.ml.models.train_lsm import TrainLSM

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float32)


# Exploratory Data Analysis (EDA)
def perform_eda(data, path):
    print("\nPerforming Exploratory Data Analysis...")
    os.makedirs(f'{path}/figures/eda/', exist_ok=True)

    # 1. Basic Statistics
    print("\nBasic Statistics:")
    stats_df = pd.DataFrame(data['inputs']['all']).describe()
    print(stats_df)
    stats_df.to_csv(f'{path}/figures/eda/basic_statistics.csv')

    # 2. Distribution of Labels
    plt.figure(figsize=(12, 6))
    sns.countplot(x=data['labels']['all'])
    plt.title('Distribution of Labels')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/eda/label_distribution.png')
    plt.close()

    # 3. Distribution of Batches
    plt.figure(figsize=(12, 6))
    sns.countplot(x=data['batches']['all'])
    plt.title('Distribution of Batches')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/eda/batch_distribution.png')
    plt.close()

    # 4. Correlation Analysis
    # print("\nComputing correlation matrix...")
    # corr_matrix = pd.DataFrame(data['inputs']['all']).corr()
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    # plt.title('Feature Correlation Matrix')
    # plt.tight_layout()
    # plt.savefig(f'{path}/figures/eda/correlation_matrix.png')
    # plt.close()

    # 5. PCA Analysis
    print("\nPerforming PCA analysis...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data['inputs']['all'])

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                          c=pd.factorize(data['labels']['all'])[0],
                          cmap='tab20')
    plt.title('PCA Visualization')
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.colorbar(scatter)
    plt.savefig(f'{path}/figures/eda/pca_visualization.png')
    plt.close()

    # 6. Missing Values Analysis
    # print("\nAnalyzing missing values...")
    # missing_values = pd.DataFrame(data['inputs']['all']).isnull().sum()
    # missing_percentage = (missing_values / len(data['inputs']['all'])) * 100
    # plt.figure(figsize=(12, 6))
    # missing_percentage[missing_percentage > 0].plot(kind='bar')
    # plt.title('Percentage of Missing Values by Feature')
    # plt.xlabel('Features')
    # plt.ylabel('Percentage of Missing Values')
    # plt.tight_layout()
    # plt.savefig(f'{path}/figures/eda/missing_values.png')
    # plt.close()
    # 7. Save Summary Statistics

    # Make a figure of the distribution of the features means
    plt.figure(figsize=(12, 6))
    sns.histplot(data['inputs']['all'].to_numpy().mean())
    plt.title('Distribution of Features Means')
    plt.xlabel('Feature Means')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/eda/feature_means_distribution.png')

    # same thing but without the zeros
    plt.figure(figsize=(12, 6))
    sns.histplot(data['inputs']['all'].to_numpy().mean()[data['inputs']['all'].to_numpy().mean() != 0])
    plt.title('Distribution of Features Means (Without Zeros)')
    plt.xlabel('Feature Means')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/eda/feature_means_distribution_without_zeros.png')

    plt.close()

    # Boxplot but per class
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=data['labels']['all'], y=data['inputs']['all'].to_numpy().mean())
    plt.title('Distribution of Features Means per Class')
    plt.xlabel('Class')
    plt.ylabel('Feature Means')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/eda/feature_means_boxplot_per_class.png')
    plt.close()

    # Violin plot but per class
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=data['labels']['all'], y=data['inputs']['all'].to_numpy().mean())
    plt.title('Distribution of Features Means per Class')
    plt.xlabel('Class')
    plt.ylabel('Feature Means')
    plt.tight_layout()  
    plt.savefig(f'{path}/figures/eda/feature_means_violinplot_per_class.png')
    plt.close()

    # Boxplot but per batch
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=data['batches']['all'], y=data['inputs']['all'].to_numpy().mean())
    plt.title('Distribution of Features Means per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Feature Means')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/eda/feature_means_boxplot_per_batch.png')
    plt.close()

    # Violin plot but per batch
    plt.figure(figsize=(12, 6))
    sns.violinplot(x=data['batches']['all'], y=data['inputs']['all'].to_numpy().mean())
    plt.title('Distribution of Features Means per Batch (Violin Plot)')
    plt.xlabel('Batch')
    plt.ylabel('Feature Means')
    plt.tight_layout()
    # Flatten to get all values
    values = data['inputs']['all'].to_numpy().flatten()

    # Count zeros and non-zeros
    zero_count = np.sum(values == 0)
    nonzero_count = np.sum(values != 0)

    # Barplot
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["Zero", "Non-Zero"], y=[zero_count, nonzero_count])
    plt.title("Feature Values: Zeros vs Non-Zeros")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/eda/feature_zeros_vs_non_zeros.png")

    # Flatten the data
    values = data['inputs']['all'].to_numpy().flatten()

    # Create a DataFrame separating zero and non-zero values
    df = pd.DataFrame({
        "Value": values,
        "Type": np.where(values == 0, "Zero", "Non-Zero")
    })

    # Filter to exclude exact zeros for meaningful distribution visualization
    # nonzero_df = df[df["Type"] == "Non-Zero"]

    # Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Type", y="Value", data=df[df["Type"] == "Non-Zero"])  # only plot non-zero values
    plt.title("Boxplot of Feature Values (Non-Zero Only)")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/eda/boxplot_feature_non_zeros.png")

    # Violin plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(x="Type", y="Value", data=df[df["Type"] == "Non-Zero"], cut=0)
    plt.title("Violin Plot of Feature Values (Non-Zero Only)")
    plt.tight_layout()
    plt.savefig(f"{path}/figures/eda/violinplot_feature_non_zeros.png")

    summary = {
        'n_samples': len(data['inputs']['all']),
        'n_features': data['inputs']['all'].shape[1],
        'n_classes': len(np.unique(data['labels']['all'])),
        'n_batches': len(np.unique(data['batches']['all'])),
        'pca_explained_variance': pca.explained_variance_ratio_.sum(),
        'mean': data['inputs']['all'].to_numpy().flatten().mean(),
        'std': data['inputs']['all'].to_numpy().flatten().std(),
        'min': data['inputs']['all'].to_numpy().flatten().min(),
        'max': data['inputs']['all'].to_numpy().flatten().max(),
    }

    pd.DataFrame([summary]).to_csv(f'{path}/figures/eda/summary_statistics.csv')
    print("\nEDA completed. Results saved in:", f'{path}/figures/eda/')


def create_objective(train, args):
    def objective(trial):
        if args.model_name == 'RF':
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'n_aug': trial.suggest_int('n_aug', 0, 1),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'max_features': trial.suggest_int("max_features", 1, 100),
                'min_samples_split': trial.suggest_int("min_samples_split", 2, 10),
                'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
                'n_estimators': trial.suggest_int("n_estimators", 100, 1000),
                'criterion': trial.suggest_categorical("criterion", ['gini', 'entropy']),
                'oob_score': trial.suggest_categorical("oob_score", [True, False]),
                'class_weight': trial.suggest_categorical("class_weight", ['balanced']),
                'scaler': trial.suggest_categorical("scaler", ['minmax2', 'l2', 'l1', 'zscore']),
            }
        elif args.model_name == 'NB':
            params = {
                'n_aug': trial.suggest_int('n_aug', 0, 5),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'scaler': trial.suggest_categorical("scaler", ['minmax2', 'l2', 'l1', 'zscore']),
            }
        elif args.model_name == 'linsvc':
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'n_aug': trial.suggest_int('n_aug', 0, 1),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'tol': trial.suggest_float('tol', 1e-4, 10, log=True),
                'max_iter': trial.suggest_int('max_iter', 3000, 10000),
                'penalty': trial.suggest_categorical('penalty', ['l2']),
                'loss': trial.suggest_categorical('loss', ['hinge', 'squared_hinge']),
                'C': trial.suggest_float('C', 1e-5, 1),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
                'scaler': trial.suggest_categorical("scaler", ['minmax2', 'l2', 'l1', 'zscore']),
            }
        elif args.model_name == 'knn':
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'n_aug': trial.suggest_int('n_aug', 0, 1),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'n_neighbors': trial.suggest_int('n_neighbors', 0, 10),
                'weights': trial.suggest_categorical('weights', ['uniform', 'balanced']),
                'scaler': trial.suggest_categorical("scaler", ['minmax2', 'l2', 'l1', 'zscore']),
            }
        elif args.model_name == 'svclinear':
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'n_aug': trial.suggest_int('n_aug', 0, 1),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'tol': trial.suggest_float('tol', 1e-4, 10, log=True),
                'max_iter': trial.suggest_int('max_iter', 1, 10000),
                'C': trial.suggest_float('C', 1e-3, 10000),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
                'kernel': trial.suggest_categorical('kernel', ['linear']),
                'probability': trial.suggest_categorical('probability', [True]),
                'scaler': trial.suggest_categorical("scaler", ['minmax2', 'l2', 'l1', 'zscore']),
            }
        elif args.model_name == 'LDA':
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'n_aug': trial.suggest_int('n_aug', 0, 5),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'scaler': trial.suggest_categorical("scaler", ['minmax2', 'l2', 'l1', 'zscore']),
            }
        elif args.model_name == 'logreg':
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'n_aug': trial.suggest_int('n_aug', 0, 1),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'max_iter': trial.suggest_int('max_iter', 1, 100),
                'C': trial.suggest_float('C', 1e-3, 20000),
                'solver': trial.suggest_categorical('solver', ['saga']),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
                'scaler': trial.suggest_categorical("scaler", ['minmax2', 'l2', 'l1', 'zscore']),
            }
        elif args.model_name == 'xgboost':
            print('XGBOOST')
            if args.n_features == -1:
                params = {
                    'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                    'p': trial.suggest_float('p', 0.0, 0.5),
                    'g': trial.suggest_float('g', 0.0, 0.5),
                    'max_depth': trial.suggest_int('max_depth', 4, 5),
                    'early_stopping_rounds': trial.suggest_float('early_stopping_rounds', 10, 20),
                    'n_estimators': trial.suggest_int('n_estimators', 1, 1000),
                    'scaler': trial.suggest_categorical("scaler", ['zscore', 'minmax']),
                }
            else:
                params = {
                    'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                    'p': trial.suggest_float('p', 0.0, 0.5),
                    'g': trial.suggest_float('g', 0.0, 0.5),
                    'max_depth': trial.suggest_int('max_depth', 4, 5),
                    'early_stopping_rounds': trial.suggest_float('early_stopping_rounds', 10, 20),
                    'n_estimators': trial.suggest_int('n_estimators', 1, 2),
                    'scaler': trial.suggest_categorical("scaler", ['minmax', 'minmax2', 'robust', 'standard', 'minmax_per_batch', 'standard_per_batch', 'robust_per_batch']),
                }

        elif args.model_name == 'bernn':
            print(f'BERNN: {args.dloss}')
            if args.n_features == 100:
                params = {
                    'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                    'p': trial.suggest_float('p', 0.0, 0.5),
                    'g': trial.suggest_float('g', 0.0, 0.5),
                    'layer1': trial.suggest_int('layer1', 32, 64),
                    'layer2': trial.suggest_int('layer2', 16, 32),
                    # 'layer3': trial.suggest_int('layer3', 512, 1024),
                    # 'layer4': trial.suggest_int('layer4', 256, 512),
                    'margin': trial.suggest_float('margin', 0.0, 0.2),
                    'smoothing': trial.suggest_float('smoothing', 0.0, 0.2),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                    'nu': trial.suggest_float('nu', 1e-4, 1e2),
                    'lr': trial.suggest_float('lr', 1e-4, 1e-3),
                    'wd': trial.suggest_float('wd', 1e-8, 1e-5),
                    'scaler': trial.suggest_categorical("scaler", ['minmax']),
                    'gamma': trial.suggest_float('gamma', 1e-3, 1e-1),
                    'warmup': trial.suggest_int('warmup', 1, 100),
                    # 'beta': trial.suggest_float('beta', 1e-2, 1e2),
                    # 'zeta': trial.suggest_float('zeta', 1e-2, 1e2),
                    'reg_entropy': trial.suggest_float('reg_entropy', 1e-8, 1e-2),
                    'l1': trial.suggest_float('l1', 1e-8, 1e-5),
                    'prune_threshold': trial.suggest_float('prune_threshold', 1e-3, 3e-3),
                }
            if args.n_features == 1000:
                params = {
                    'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                    'p': trial.suggest_float('p', 0.0, 0.5),
                    'g': trial.suggest_float('g', 0.0, 0.5),
                    'layer1': trial.suggest_int('layer1', 32, 128),
                    # 'layer2': trial.suggest_int('layer2', 128, 256),
                    # 'layer3': trial.suggest_int('layer3', 512, 1024),
                    # 'layer4': trial.suggest_int('layer4', 256, 512),
                    'margin': trial.suggest_float('margin', 0.0, 0.2),
                    'smoothing': trial.suggest_float('smoothing', 0.0, 0.2),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                    'nu': trial.suggest_float('nu', 1e-4, 1e2),
                    'lr': trial.suggest_float('lr', 1e-5, 1e-3),
                    'wd': trial.suggest_float('wd', 1e-8, 1e-5),
                    'scaler': trial.suggest_categorical("scaler", ['minmax']),
                    'gamma': trial.suggest_float('gamma', 1e-2, 1e2),
                    'warmup': trial.suggest_int('warmup', 1, 2),
                    # 'beta': trial.suggest_float('beta', 1e-2, 1e2),
                    # 'zeta': trial.suggest_float('zeta', 1e-2, 1e2),
                    'reg_entropy': trial.suggest_float('reg_entropy', 1e-8, 1e-2),
                    'l1': trial.suggest_float('l1', 1e-8, 1e-5),
                    'prune_threshold': trial.suggest_float('prune_threshold', 1e-3, 3e-3),
                }
            else:
                params = {
                    'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                    # 'p': trial.suggest_float('p', 0.0, 0.5),
                    # 'g': trial.suggest_float('g', 0.0, 0.5),
                    'margin': trial.suggest_float('margin', 0.0, 0.2),
                    'smoothing': trial.suggest_float('smoothing', 0.0, 0.2),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                    'nu': trial.suggest_float('nu', 1e-2, 1e2, log=True),
                    'lr': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
                    'wd': trial.suggest_float('wd', 1e-8, 1e-5, log=True),
                    'scaler': trial.suggest_categorical("scaler", ['minmax', 'robust', 'standard', 'minmax_per_batch', 'standard_per_batch', 'robust_per_batch']),
                    'warmup': trial.suggest_int('warmup', 10, args.max_warmup, log=True),
                    # 'reg_entropy': trial.suggest_float('reg_entropy', 1e-8, 1e-2),
                    'l1': trial.suggest_float('l1', 1e-8, 1e-5),
                    # 'prune_threshold': trial.suggest_float('prune_threshold', 1e-3, 3e-3),
                    # 'tied_weights': trial.suggest_categorical('tied_weights', [0, 1]),
                    'add_noise': trial.suggest_categorical('add_noise', [0, 1]),
                    'use_l1': trial.suggest_categorical('use_l1', [0, 1]),
                    'use_dropout': trial.suggest_categorical('use_dropout', [0, 1]),
                }
            for i, layer in enumerate(args.ae_layers_max_neurons):
                params[f'layer{i+1}'] = trial.suggest_int(f'layer{i+1}', 100, layer)
            # Some hyperparameters are not always required. They are set to a default value in Train.train()
            if args.dloss in ['revTriplet', 'revDANN', 'DANN', 'inverseTriplet', 'normae']:
                # gamma = 0 will ensure DANN is not learned
                params['gamma'] = trial.suggest_float('gamma', 1e-8, 1e0, log=True)
            if args.variational:
                # beta = 0 because useless outside a variational autoencoder
                params['beta'] = trial.suggest_float('beta', 1e-8, 1e0, log=True)
            if args.zinb:
                # zeta = 0 because useless outside a zinb autoencoder
                params['zeta'] = trial.suggest_float('zeta', 1e-2, 1e2, log=True)
            if 'threshold' not in params:
                params['threshold'] = 0.  # TODO this should ne handled in bernn
        elif args.model_name == 'xgboostda':
            print('XGBOOST DASK')
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                'early_stopping_rounds': trial.suggest_float('early_stopping_rounds', 10, 20),
                'n_estimators': trial.suggest_int('n_estimators', 100, 150),
                'scaler': trial.suggest_categorical("scaler", ['minmax2']),
            }
        elif args.model_name == 'xgboostex':
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'n_aug': trial.suggest_int('n_aug', 0, 1),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'max_depth': trial.suggest_int('max_depth', 5, 9),
                'early_stopping_rounds': trial.suggest_float('early_stopping_rounds', 10, 20),
                'n_estimators': trial.suggest_int('n_estimators', 300, 350),
                'scaler': trial.suggest_categorical("scaler", ['minmax2']),
            }
            # Some hyperparameters are not always required. 
        return train.train(params)
    return objective


def make_args(args, batches_to_keep, concs):
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
    concs = args.concs.split(',')
    new_cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    args.exp_name = f'{"-".join(batches_to_keep)}_binary{args.binary}_{args.n_features}' \
                    f'_gkf{args.groupkfold}_ovr{args.ovr}_{new_cropings}_{"_".join(concs)}'
    if args.rec_prototype == 1 and args.use_mapping == 1:
        print('Using prototype: mapping turned off')
        args.use_mapping = 0
    args.ae_layers_max_neurons = [int(x) for x in args.ae_layers_max_neurons.split(',')]
    
    if args.normalize and args.use_sigmoid:
        print('Using normalization: sigmoid final activation cannot be used')
        args.use_sigmoid = 0

    return args


def get_batches_infos():
    # TODO: Get from folder or database
    batch_dates = [
        "BPatients-03-14-2025", "cultures_pures", "B15-06-29-2024",
        "B14-06-10-2024", "B13-06-05-2024", "B12-05-31-2024", "B11-05-24-2024",
        "B10-05-03-2024", "B9-04-22-2024", "B8-04-15-2024",
        'B7-04-03-2024', 'B6-03-29-2024', 'B5-03-13-2024',
        'B4-03-01-2024', 'B3-02-29-2024', 'B2-02-21-2024',
        'B1-02-02-2024'
    ]
    batches_to_keep = [
        "BPatients-03-14-2025", "cultures_pures", "b15-06-29-2024",
        "b14-06-10-2024", "b13-06-05-2024", "b12-05-31-2024", "b11-05-24-2024",
        "b10-05-03-2024", "b9-04-22-2024", "b8-04-15-2024",
        'b7-04-03-2024', 'b6-03-29-2024', 'b5-03-13-2024',
        'b4-03-01-2024', 'b3-02-29-2024', 'b2-02-21-2024',
        'b1-02-02-2024'
    ]
    batch_dates = [x.split('-')[0] for x in batch_dates]
    batches_to_keep = [x.split('-')[0] for x in batches_to_keep]
    return batch_dates, batches_to_keep


def keep_some_batches(data, batches_to_keep):
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

    return data


def keep_some_concs(data, concs):
    mask2 = [True if x in concs else False for x in data['concs']['all']]
    data['inputs']['all'] = data['inputs']['all'].iloc[mask2]
    data['labels']['all'] = data['labels']['all'][mask2]
    data['batches']['all'] = data['batches']['all'][mask2]
    data['batches_labels']['all'] = data['batches_labels']['all'][mask2]
    data['names']['all'] = data['names']['all'][mask2]
    data['orders']['all'] = data['orders']['all'][mask2]
    data['cats']['all'] = data['cats']['all'][mask2]
    data['urines']['all'] = data['urines']['all'][mask2]
    data['manips']['all'] = data['manips']['all'][mask2]
    data['concs']['all'] = data['concs']['all'][mask2]

    # TODO Assert that no concentration is left that should not
    try:
        assert len(np.unique(data['concs']['all'])) == len(concs)
    except Exception as e:
        print("Problem at assert len(np.unique(data['concs']['all'])) == len(concs)", e)
        raise ValueError(f'Problem with concentrations: Should have: {concs} but got {np.unique(data["concs"]["all"])}')

    return data


def get_args(batches_to_keep):
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--features_file', type=str, default='mutual_info_classif_scores.csv')
    parser.add_argument('--n_epochs', type=int, default=1000)
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
    parser.add_argument('--bs', type=int, default=64)
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
    parser.add_argument('--mode', type=str, default='logaddinloop')
    parser.add_argument('--features_selection', type=str, default='none')
    parser.add_argument('--concs', type=str, default='na,h,l')
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
    parser.add_argument("--fp", type=str, default='float32', help='float16, float32, float64')
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--max_bin", type=int, default=256)
    parser.add_argument("--sparse_matrix", type=int, default=0)
    parser.add_argument("--log_plots", type=int, default=1)
    parser.add_argument("--log_metrics", type=int, default=1)
    parser.add_argument("--log_neptune", type=int, default=1)
    parser.add_argument("--prune_threshold", type=float, default=0)
    parser.add_argument("--dloss", type=str, default='DANN')
    parser.add_argument("--warmup_after_warmup", type=int, default=1)
    parser.add_argument("--train_after_warmup", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--fast_hparams_optim", type=int, default=0)
    parser.add_argument('--patient_bact', type=str, default='', help='put nothing to analyse all bacteria')
    parser.add_argument('--random_recs', type=int, default=0)
    parser.add_argument('--rec_prototype', type=int, default=1)
    parser.add_argument('--max_norm', type=float, default=0.)
    parser.add_argument('--max_warmup', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--add_noise', type=int, default=1)
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--use_mapping', type=int, default=0)
    parser.add_argument('--use_sigmoid', type=int, default=0)
    parser.add_argument('--use_threshold', type=int, default=0)
    # parser.add_argument('--use_dropout', type=int, default=0)
    parser.add_argument('--use_smoothing', type=int, default=0)
    parser.add_argument('--variational', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=50, help='Use early stopping during training')
    parser.add_argument('--early_warmup_stop', type=int, default=10, help='Use early stopping during warmup')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the classifier')
    parser.add_argument('--ae_layers_max_neurons', type=str, default='1000', help='Maximum number of neurons in the AE layers')
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='Learning rate scheduler type')
    parser.add_argument('--xgboost_features', type=int, default=0, help='Use xgboost top parameters')
    parser.add_argument('--clip_val', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--kan', type=int, default=0, help='Use KAN during training')
    parser.add_argument('--classif_loss', type=str, default='ce', help='Classification loss function. [ce, triplet]')
    parser.add_argument('--rec_loss', type=str, default='l1', help='Reconstruction loss function. [l1, mse]')
    parser.add_argument('--min_features_importance', type=int, default=0, help='Minimum feature importance to consider')

    args = parser.parse_args()

    concs = args.concs.split(',')
    args = make_args(args, batches_to_keep, concs)

    return args


def get_path(args, exp):
    path = f'resources/bacteries_2024/matrices/mz{args.mz}/rt{args.rt}/' \
        f'mzp{args.mzp}/rtp{args.rtp}/thr{args.threshold}/{args.spd}spd/' \
        f'ms{args.ms_level}/combat{args.combat}/shift{args.shift}/none/' \
        f'{args.mode}/{args.features_selection}/{exp}'
    results_path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/{args.spd}spd/thr{args.threshold}/' \
        f'{args.train_on}/{args.exp_name}/{args.model_name}/'
    return path, results_path


def change_data_type(data, args):
    if args.fp == 'float16':
        for g in list(data['inputs'].keys()):
            data['inputs'][g] = data['inputs'][g].astype(np.float16)
    elif args.fp == 'float32':
        for g in list(data['inputs'].keys()):
            data['inputs'][g] = data['inputs'][g].astype(np.float32)
    elif args.fp == 'float64':
        for g in list(data['inputs'].keys()):
            data['inputs'][g] = data['inputs'][g].astype(np.float64)
    return data


def get_model(args):
    # Initialize the model based on args.model_name
    if args.model_name == 'RF':
        cfr = RandomForestClassifier
    elif args.model_name == 'NB':
        cfr = GaussianNB
    elif args.model_name == 'linsvc':
        cfr = sklearn.svm.LinearSVC
    elif args.model_name == 'knn':
        cfr = sklearn.neighbors.KNeighborsClassifier
    elif args.model_name == 'svclinear':
        cfr = sklearn.svm.SVC
    elif args.model_name == 'LDA':
        cfr = LDA
    elif args.model_name == 'logreg':
        cfr = sklearn.linear_model.LogisticRegression
    elif args.model_name == 'xgboost':
        print('XGBOOST')
        Train = Train_xgboost
        cfr = xgboost.XGBClassifier
    elif args.model_name == 'bernn':
        if args.dloss == 'DANN':
            print('BERNN: DANN')
        elif args.dloss == 'no':
            print('BERNN: AE')
        elif args.dloss == 'inverseTriplet':
            print('BERNN: inverseTriplet')
        else:
            raise ValueError(f'Invalid loss function: {args.dloss}')
        args.kan = 0
        Train = Train_bernn
        cfr = TrainAEClassifierHoldout
    elif args.model_name == 'xgboostda':
        print('XGBOOST DASK')
        from train_xgboost_dask import Train
        cfr = xgboost.XGBClassifier
    elif args.model_name == 'xgboostex':
        from train_xgboost_extmem import Train
        cfr = xgboost.XGBClassifier
    elif args.model_name == 'lsm':
        print('LSM: LargeSpectralBERT')
        Train = TrainLSM
        cfr = None  # Not used for LSM, handled inside TrainLSM
    else:
        raise ValueError(f'Invalid model name: {args.model_name}')
    return cfr, Train, args

def select_xgboost_features(data, uniques, concs, batch_dates, scaler, args):
    # Select the top n_features based on importance
    print(f'Selecting top {args.n_features} features using XGBoost results')
    path = (
        f"results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/"
        f"{args.spd}spd/thr0.0/all/"
        f"{'-'.join(batch_dates)}_binary{args.binary}_-1_"
        f"gkf{args.groupkfold}_ovr{args.ovr}_"
        f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}_"
        f"{'_'.join(concs)}/xgboost/ords_filtered/"
        f"xgboost_feature_importance_{scaler}.csv"
    )
    feature_importance = pd.read_csv(path)
    if args.n_features > 0:
        top_features = feature_importance.nlargest(args.n_features, 'importance')['feature'].values
    else:
        top_features = feature_importance.nlargest(feature_importance.shape[0], 'importance')['feature'].values
    # Filter the data to keep only the top features
    print(f"Selected features: {top_features.shape[0]} features from initial {len(data['inputs']['all'])}")
    data['inputs'] = {k: v.loc[:, top_features] for k, v in data['inputs'].items()}

    return data


if __name__ == '__main__':
    batch_dates, batches_to_keep = get_batches_infos()
    args = get_args(batches_to_keep)
    concs = args.concs.split(',')
    cropings = "mz0-10000rt0-320"
    exp = f'all_{"-".join(batch_dates)}_gkf{args.groupkfold}_{cropings}_5splits'
    path, results_path = get_path(args, exp)
    data, uniques = get_data_all(path, args)
    # if args.xgboost_features:
    #     data = select_xgboost_features(data, uniques, concs, batches_to_keep, 'zscore', args)

    # Perform EDA
    # perform_eda(data, path)
    data = change_data_type(data, args)
    data = keep_some_concs(data, concs)
    # data = keep_some_batches(data, batches_to_keep)

    cfr, Train, args = get_model(args)
    if args.model_name == 'lsm':
        train = Train(name="inputs", model=None, data=data, uniques=uniques,
                      log_path=results_path, args=args, log_metrics=args.log_metrics,
                      logger=None, log_neptune=args.log_neptune, mlops='None')
    else:
        train = Train(name="inputs", model=cfr, data=data, uniques=uniques,
                      log_path=results_path, args=args, log_metrics=args.log_metrics,
                      logger=None, log_neptune=args.log_neptune, mlops='None')

    # Create and run the Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)  # <--- Seed Optuna's sampler
    )
    study.optimize(create_objective(train, args), n_trials=30)

    # Save optimization results and plots
    os.makedirs(f'{results_path}/figures/optuna/', exist_ok=True)

    # Plot optimization history
    fig = plot_optimization_history(study)
    fig.write_image(f'{results_path}/figures/optuna/optimization_history.png')

    # Plot parameter importances
    fig = plot_param_importances(study)
    fig.write_image(f'{results_path}/figures/optuna/param_importances.png')

    # Plot parallel coordinate
    fig = plot_parallel_coordinate(study)
    fig.write_image(f'{results_path}/figures/optuna/parallel_coordinate.png')

    # Save best parameters
    best_params = study.best_params
    print("Best parameters:", best_params)
    print("Best value:", study.best_value)
