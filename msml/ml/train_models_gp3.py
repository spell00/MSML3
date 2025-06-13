import os
import xgboost
import numpy as np
import random
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
from .dataset import get_data_all
from sklearn.naive_bayes import GaussianNB
from .train_xgboost import Train_xgboost
from .train_bernn import Train_bernn
from bernn import TrainAEClassifierHoldout

NEPTUNE_API_TOKEN = os.environ.get('NEPTUNE_API_TOKEN')
NEPTUNE_PROJECT_NAME = "MSML-Bacteria"
NEPTUNE_MODEL_NAME = 'MSMLBAC-'

warnings.filterwarnings("ignore")

random.seed(42)
# torch.manual_seed(42)
np.random.seed(42)


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
    nonzero_df = df[df["Type"] == "Non-Zero"]

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
            params = {
                'threshold': trial.suggest_float('threshold', 0.0, 0.5),
                'p': trial.suggest_float('p', 0.0, 0.5),
                'g': trial.suggest_float('g', 0.0, 0.5),
                'max_depth': trial.suggest_int('max_depth', 4, 5),
                'early_stopping_rounds': trial.suggest_float('early_stopping_rounds', 10, 20),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'scaler': trial.suggest_categorical("scaler", ['zscore', 'minmax2']),
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
                    'lr': trial.suggest_float('lr', 1e-4, 1e-2),
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
                    'p': trial.suggest_float('p', 0.0, 0.5),
                    'g': trial.suggest_float('g', 0.0, 0.5),
                    'layer1': trial.suggest_int('layer1', 1000, 10000),
                    # 'layer2': trial.suggest_int('layer2', 1024, 2048),
                    # 'layer3': trial.suggest_int('layer3', 512, 1024),
                    # 'layer4': trial.suggest_int('layer4', 256, 512),
                    'margin': trial.suggest_float('margin', 0.0, 0.2),
                    'smoothing': trial.suggest_float('smoothing', 0.0, 0.2),
                    'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                    'nu': trial.suggest_float('nu', 1e-4, 1e2),
                    'lr': trial.suggest_float('lr', 1e-5, 1e-2),
                    'wd': trial.suggest_float('wd', 1e-8, 1e-5),
                    'scaler': trial.suggest_categorical("scaler", ['minmax', 'minmax2', 'zscore']),
                    'gamma': trial.suggest_float('gamma', 1e-2, 1e2),
                    'warmup': trial.suggest_int('warmup', 1, 250),
                    # 'beta': trial.suggest_float('beta', 1e-2, 1e2),
                    # 'zeta': trial.suggest_float('zeta', 1e-2, 1e2),
                    'reg_entropy': trial.suggest_float('reg_entropy', 1e-8, 1e-2),
                    'l1': trial.suggest_float('l1', 1e-8, 1e-5),
                    'prune_threshold': trial.suggest_float('prune_threshold', 1e-3, 3e-3),
                }

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
    return args


def get_batches_infos():
    # TODO: Get from folder or database
    batch_dates = [
        "BPatients-03-14-2025", "B15-06-29-2024",
        "B14-06-10-2024", "B13-06-05-2024", "B12-05-31-2024", "B11-05-24-2024",
        "B10-05-03-2024", "B9-04-22-2024", "B8-04-15-2024",
        'B7-04-03-2024', 'B6-03-29-2024', 'B5-03-13-2024',
        'B4-03-01-2024', 'B3-02-29-2024', 'B2-02-21-2024',
        'B1-02-02-2024'
    ]
    batches_to_keep = [
        "BPatients-03-14-2025", "b15-06-29-2024",
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
    parser.add_argument('--n_epochs', type=int, default=3)
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
    parser.add_argument("--fp", type=str, default=32)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--max_bin", type=int, default=256)
    parser.add_argument("--sparse_matrix", type=int, default=0)
    parser.add_argument("--log_plots", type=int, default=1)
    parser.add_argument("--prune_threshold", type=float, default=0)
    parser.add_argument("--dloss", type=str, default='DANN')
    parser.add_argument("--warmup_after_warmup", type=int, default=1)
    parser.add_argument("--train_after_warmup", type=int, default=1)
    parser.add_argument("--test", type=int, default=0)

    args = parser.parse_args()

    concs = args.concs.split(',')
    args = make_args(args, batches_to_keep, concs)

    return args


def get_path(args, exp):
    path = f'resources/bacteries_2024/matrices/mz{args.mz}/rt{args.rt}/' \
        f'mzp{args.mzp}/rtp{args.rtp}/thr{args.threshold}/{args.spd}spd/' \
        f'ms{args.ms_level}/combat{args.combat}/shift{args.shift}/none/' \
        f'log{args.log}/{args.features_selection}/{exp}'
    results_path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/{args.spd}spd/thr{args.threshold}/' \
        f'{args.train_on}/{args.exp_name}/{args.model_name}/'
    return path, results_path


def change_data_type(data, args):
    if args.fp == 'float16':
        data['inputs']['all'] = data['inputs']['all'].astype(np.float16)
    elif args.fp == 'float32':
        data['inputs']['all'] = data['inputs']['all'].astype(np.float32)
    elif args.fp == 'float64':
        data['inputs']['all'] = data['inputs']['all'].astype(np.float64)
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
    else:
        raise ValueError(f'Invalid model name: {args.model_name}')
    return cfr, Train, args


if __name__ == '__main__':
    batch_dates, batches_to_keep = get_batches_infos()
    args = get_args(batches_to_keep)
    concs = args.concs.split(',')
    batch_dates, batches_to_keep = get_batches_infos()
    cropings = "mz0-10000rt0-320"
    exp = f'all_{"-".join(batch_dates)}_gkf{args.groupkfold}_{cropings}_5splits'  
    path, results_path = get_path(args, exp)
    data, uniques = get_data_all(path, args)

    # Perform EDA
    # perform_eda(data, path)
    data = change_data_type(data, args)
    data = keep_some_concs(data, concs)
    # data = keep_some_batches(data, batches_to_keep)

    cfr, Train, args = get_model(args)
    train = Train(name="inputs", model=cfr, data=data, uniques=uniques,
                  log_path=results_path, args=args, log_metrics=True,
                  logger=None, log_neptune=True, mlops='None')

    # Create and run the Optuna study
    study = optuna.create_study(direction='minimize')
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
