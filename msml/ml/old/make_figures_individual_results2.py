import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC

bacteria_cols = ['blanc','aba','api','aur','cal','cfam','cfr','cgl',
                'cko','cpa','ecl','eco','efa','efc','kae','kox','kpn',
                'mmo','pae','pmi','pre','pst','pvu','sag','sau','sha',
                'slu','sma','ssa']

# Calculate tpr by bacterium
def calculate_tpr_per_bact(df, bacteria_cols):
    true_positives = {}
    for bacterium in bacteria_cols:
        positives = df[df['labels'] == df['preds']][df['labels'] == bacterium].shape[0]
        total_predictions = (df['labels'] == bacterium).sum()
        true_positives[bacterium] = positives / total_predictions
    return true_positives

# Calculate tpr by batch
def calculate_tpr_per_batch(df, batches):
    true_positives = {}
    for batch in batches:
        corrects = df[df['labels'] == df['preds']][df['batches'] == batch].shape[0]
        total = (df['batches'] == batch).sum()
        true_positives[batch] = corrects / total
    return true_positives

def calculate_acc_per_bact(df, bacteria_cols):
    mcc = {}
    for bacterium in bacteria_cols:
        tmp = df.copy()
        tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        mcc[bacterium] = ACC(tmp['labels'], tmp['preds'])
    return mcc

def calculate_acc_per_batch(df, batches):
    mcc = {}
    for batch in batches:
        tmp = df.copy().iloc[df[df['batches'] == batch].index]
        mcc[batch] = ACC(tmp['labels'], tmp['preds'])
    return mcc


def calculate_precision_per_bact(df, bacteria_cols):
    precision = {}
    for bacterium in bacteria_cols:
        tmp = df.copy()
        tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        corrects = df[df['labels'] == df['preds']][df['labels'] == bacterium].shape[0]
        total = df[df['preds'] == bacterium].shape[0]
        precision[bacterium] = corrects / total
    return precision

def calculate_precision_per_batch(df, batches):
    precision = {}
    for batch in batches:
        tmp = df.copy().iloc[df[df['batches'] == batch].index]
        corrects = tmp[tmp['labels'] == tmp['preds']].shape[0]
        total = tmp.shape[0]
        precision[batch] = corrects / total
    return precision

def calculate_mcc_per_bact(df, bacteria_cols):
    mcc = {}
    for bacterium in bacteria_cols:
        tmp = df.copy()
        tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        mcc[bacterium] = MCC(tmp['labels'], tmp['preds'])
    return mcc

def calculate_mcc_per_batch(df, batches):
    mcc = {}
    for batch in batches:
        tmp = df.copy().iloc[df[df['batches'] == batch].index]
        mcc[batch] = MCC(tmp['labels'], tmp['preds'])
    return mcc

def make_graph(valid_metrics, test_metrics, metric, category, cols, title, path):
    # Prepare data for plotting
    metric_df = pd.DataFrame({
        category: cols,
        'Valid': [valid_metrics[b] for b in cols],
        'Test': [test_metrics[b] for b in cols]
    })

    metric_melted_df = metric_df.melt(id_vars=category, var_name='Group', value_name=metric)

    # Plot metric by bacterium
    plt.figure(figsize=(14, 8))
    sns.barplot(x=category, y=metric, hue='Group', data=metric_melted_df)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.savefig(f'test_{category}.png')

def make_total_graph_total_batch(valid_metrics, test_metrics, metric, title, path):
    # Prepare data for plotting
    metric_df = pd.DataFrame({
        'Batch': valid_metrics.keys(),
        'Valid': [valid_metrics[b] for b in valid_metrics.keys()],
        'Test': [test_metrics[b] for b in test_metrics.keys()]
    })

    metric_melted_df = metric_df.melt(id_vars='Batch', var_name='Group', value_name=metric)
    # Add a column Batches with only the sAME VALUE Batches
    metric_melted_df['Batches'] = 'Batches'

    # Plot metric by bacterium
    plt.figure(figsize=(6, 6))
    sns.boxplot(x='Batches', y=metric, hue='Group', data=metric_melted_df)
    # Remove x axis label
    plt.xlabel('')
    # Remove ticks
    plt.xticks([])
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)

def make_total_graph_total_bact(valid_metrics, test_metrics, metric, title, path):
    # Prepare data for plotting
    metric_df = pd.DataFrame({
        'Bacteria': bacteria_cols,
        'Valid': [valid_metrics[b] for b in bacteria_cols],
        'Test': [test_metrics[b] for b in bacteria_cols]
    })

    metric_melted_df = metric_df.melt(id_vars='Bacteria', var_name='Group', value_name=metric)
    metric_melted_df['Bacteria'] = 'Bacteria'

    # Plot metric by bacterium
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Bacteria', y=metric, hue='Group', data=metric_melted_df)
    plt.xticks(rotation=90)
    plt.title(title)
    # Remove x axis label
    plt.xlabel('')
    # Remove ticks
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(path)

def plot_bars(exp_name, args):
    # Load the datasets
    path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
            f'{args.spd}spd/thr{args.threshold}/{args.train_on}/{exp_name}/' \
            f'{args.model}/saved_models/'
    valid_df = pd.read_csv(f'{path}/{args.model}_valid_individual_results.csv', index_col=0)
    test_df = pd.read_csv(f'{path}/{args.model}_test_individual_results.csv', index_col=0)

    for metric in ['TPR', 'accuracy', 'precision', 'MCC']:
        if metric == 'TPR':
            total_valid = valid_df[valid_df['labels'] == valid_df['preds']].shape[0] / valid_df.shape[0]
            total_test = test_df[test_df['labels'] == test_df['preds']].shape[0] / test_df.shape[0]
            valid_metrics_bact = calculate_tpr_per_bact(valid_df, bacteria_cols)
            test_metrics_bact = calculate_tpr_per_bact(test_df, bacteria_cols)
            valid_metrics_batch = calculate_tpr_per_batch(valid_df, valid_df['batches'].unique())
            test_metrics_batch = calculate_tpr_per_batch(test_df, test_df['batches'].unique())
        elif metric == 'accuracy':
            total_valid = ACC(valid_df['labels'], valid_df['preds'])
            total_test = ACC(test_df['labels'], test_df['preds'])
            valid_metrics_bact = calculate_acc_per_bact(valid_df, bacteria_cols)
            test_metrics_bact = calculate_acc_per_bact(test_df, bacteria_cols)
            valid_metrics_batch = calculate_acc_per_batch(valid_df, valid_df['batches'].unique())
            test_metrics_batch = calculate_acc_per_batch(test_df, test_df['batches'].unique())
        elif metric == 'precision':
            total_valid = valid_df[valid_df['labels'] == valid_df['preds']].shape[0] / valid_df.shape[0]
            total_test = test_df[test_df['labels'] == test_df['preds']].shape[0] / test_df.shape[0]
            valid_metrics_bact = calculate_precision_per_bact(valid_df, bacteria_cols)
            test_metrics_bact = calculate_precision_per_bact(test_df, bacteria_cols)
            valid_metrics_batch = calculate_precision_per_batch(valid_df, valid_df['batches'].unique())
            test_metrics_batch = calculate_precision_per_batch(test_df, test_df['batches'].unique())
        elif metric == 'MCC':
            total_valid = MCC(valid_df['labels'], valid_df['preds'])
            total_test = MCC(test_df['labels'], test_df['preds'])
            valid_metrics_bact = calculate_mcc_per_bact(valid_df, bacteria_cols)
            test_metrics_bact = calculate_mcc_per_bact(test_df, bacteria_cols)
            valid_metrics_batch = calculate_mcc_per_batch(valid_df, valid_df['batches'].unique())
            test_metrics_batch = calculate_mcc_per_batch(test_df, test_df['batches'].unique())

        make_graph(valid_metrics_bact, test_metrics_bact, metric, 'Bacterium', bacteria_cols,
                   f'{metric} by Bacterium. Total valid: {total_valid:.2f}, test: {total_test:.2f}', 
                   f'{path}/{exp_name}_{metric}_by_bact.png'
        )

        make_graph(valid_metrics_batch, test_metrics_batch, metric, 'Batch', valid_df['batches'].unique(),
                   f'{metric} by Batch. Total valid: {total_valid:.2f}, test: {total_test:.2f}', 
                   f'{path}/{exp_name}_{metric}_by_batch.png'
        )
        make_total_graph_total_batch(valid_metrics_batch, test_metrics_batch, metric,
                                   f'{metric} by Batch. Total valid: {total_valid:.2f}, test: {total_test:.2f}', 
                                   f'{path}/{exp_name}_{metric}_by_batch_total.png'
        )
        make_total_graph_total_bact(valid_metrics_bact, test_metrics_bact, metric,
                                      f'{metric} by Bacterium. Total valid: {total_valid:.2f}, test: {total_test:.2f}', 
                                      f'{path}/{exp_name}_{metric}_by_bact_total.png'
        )


if __name__ == '__main__':
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='xgboost')
    parser.add_argument('--train_on', type=str, default='all')
    parser.add_argument('--csv_file', type=str, default='inputs.csv')
    parser.add_argument('--ovr', type=int, default=1)
    parser.add_argument('--mz', type=int, default=10)
    parser.add_argument('--rt', type=int, default=10)
    parser.add_argument('--mzp', type=int, default=10)
    parser.add_argument('--rtp', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.0)
    parser.add_argument('--spd', type=int, default=200)
    parser.add_argument('--ms_level', type=int, default=2)
    parser.add_argument('--combat', type=int, default=0)  # TODO not using this anymore
    parser.add_argument('--shift', type=int, default=0)  # TODO keep this?
    parser.add_argument('--log', type=str, default='inloop')
    parser.add_argument('--features_selection', type=str, default='mutual_info_classif')
    parser.add_argument('--concs', type=str, default='na,h')
    parser.add_argument('--n_repeats', type=int, default=5)
    parser.add_argument('--classif_loss', type=str, default='celoss')
    parser.add_argument("--min_mz", type=int, default=80)
    parser.add_argument("--max_mz", type=int, default=1200)
    parser.add_argument("--min_rt", type=int, default=140)
    parser.add_argument("--max_rt", type=int, default=320)
    parser.add_argument("--output", type=str, default='2models')
    parser.add_argument("--model", type=str, default='xgboost')
    args = parser.parse_args()

    batch_dates = [
        "B14-06-10-2024", "B13-06-05-2024", "B12-05-31-2024", "B11-05-24-2024",
        "B10-05-03-2024", "B9-04-22-2024", "B8-04-15-2024",
        'B7-04-03-2024', 'B6-03-29-2024', 'B5-03-13-2024',
        'B4-03-01-2024', 'B3-02-29-2024', 'B2-02-21-2024',
        'B1-02-02-2024', 
    ]
    batch_dates = '-'.join([x.split('-')[0].lower() for x in batch_dates])
    concs = args.concs.split(',')
    cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    if args.output != '2models':
        exp_name = f'{batch_dates}_binary{args.binary}_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_{cropings}_{"_".join(concs)}'
    else:
        exp_name = f'{batch_dates}_2models_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_remblk1_{cropings}_{"_".join(concs)}'

    plot_bars(exp_name, args)
    # plot_individual_bars(batch_dates, args)
    
