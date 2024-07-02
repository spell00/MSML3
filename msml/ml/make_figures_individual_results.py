import os
import json
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
# import MCC from sklearn
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import accuracy_score as ACC

def make_df_classes(df, name):
    new_df = {
        'models': [],
        'batches': [],
        'groups': [],
        'labels': [],
        'value': [],
    }
    unique_batches = np.unique(df['valid']['batches'])
    for group in ['valid', 'test']:
        df2 = pd.DataFrame(df[group])
        for model in np.unique(df[group]['models']):
            df1 = df2[df2["models"].str.contains(model)]
            for batch in unique_batches:
                df0 = df1[df1["batches"].str.contains(batch)]
                _, idx = np.unique(df0['classes'], return_index=True)
                unique_classes = np.array(df0['classes'])[np.sort(idx)]
                unique_labels = np.array(df0['labels'])[np.sort(idx)]
                for clas, label in zip(unique_classes, unique_labels):
                    new_classes = np.zeros(len(df0['classes']))
                    new_preds = np.zeros(len(df0['preds']))
                    new_classes[df0['classes'] == clas] = 1
                    new_preds[df0['preds'] == clas] = 1
                    new_classes[df0['classes'] != clas] = 0
                    new_preds[df0['preds'] != clas] = 0
                    new_df['models'].append(model)
                    new_df['batches'].append(batch)
                    new_df['groups'].append(group)
                    new_df['labels'].append(label)
                    if name == 'mcc':
                        new_df['value'].append(MCC(new_classes, new_preds))
                    else:
                        new_df['value'].append(ACC(new_classes, new_preds))
    new_df = pd.DataFrame(new_df)
        
    return new_df

def make_df_classes_per_batch(df):
    pass

def plot_bars(batch_dates, args):
    batch_dates = '-'.join([x.split('-')[0].lower() for x in batch_dates])
    concs = args.concs.split(',')
    cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    if args.output != '2models':
        exp_name = f'{batch_dates}_binary{args.binary}_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_{cropings}_{"_".join(concs)}'
    else:
        exp_name = f'{batch_dates}_2models_{args.n_features}' \
                        f'_gkf{args.groupkfold}_ovr{args.ovr}_remblk1_{cropings}_{"_".join(concs)}'
    for name in ['mcc', 'acc']:
        df = {
            'value': [],
            'models': [],
            'group': [],
            'batches': [],
        }
        df_all = {
            'valid': {
                'labels': [],
                'classes': [],
                'models': [],
                'preds': [],
                'batches': [],
            },
            'test': {
                'labels': [],
                'classes': [],
                'models': [],
                'preds': [],
                'batches': [],
            }
        }
        for model in ['linsvc']:
            path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
                    f'{args.spd}spd/thr{args.threshold}/{args.train_on}/{exp_name}/{model}'
            results = json.load(open(f'{path}/saved_models/best_params_inputs_{model}_values.json'))
            batches = json.load(open(f'{path}/saved_models/unique_batches_inputs_{model}.json'))
            valid_scores = pd.read_csv(open(f'{path}/saved_models/{model}_valid_individual_results.csv'))
            test_scores = pd.read_csv(open(f'{path}/saved_models/{model}_test_individual_results.csv'))

            df['value'] += results[f'valid_{name}'] + results[f'test_{name}']
            df['models'] += [model] * len(results[f'valid_{name}']) + [model] * len(results[f'test_{name}'])
            df['group'] += ['valid'] * len(results[f'valid_{name}']) + ['test'] * len(results[f'test_{name}'])
            df['batches'] += [b[0] for b in batches['valid']] + [b[0] for b in batches['test']]
            df_all['valid']['classes'] += valid_scores.loc[:, 'classes'].to_list()
            df_all['test']['classes'] += test_scores.loc[:, 'classes'].to_list()
            df_all['valid']['labels'] += valid_scores.loc[:, 'labels'].to_list()
            df_all['test']['labels'] += test_scores.loc[:, 'labels'].to_list()
            df_all['valid']['batches'] += valid_scores.loc[:, 'batches'].to_list()
            df_all['test']['batches'] += test_scores.loc[:, 'batches'].to_list()
            df_all['valid']['preds'] += valid_scores.loc[:, 'preds'].to_list()
            df_all['test']['preds'] += test_scores.loc[:, 'preds'].to_list()
            df_all['valid']['models'] += [model] * len(valid_scores.loc[:, 'preds'].to_list())
            df_all['test']['models'] += [model] * len(test_scores.loc[:, 'preds'].to_list())

        df = pd.DataFrame(df)
        df_all = make_df_classes(df_all, name)
        sns.set(rc={'figure.figsize':(12, 8)})
        sns.set_style(style='whitegrid') 
        g = sns.barplot(
            x="models",
            y='value',
            data=df,
            errorbar=None,
            hue='group',
        )
        g = sns.stripplot(
            x="models", 
            y='value', 
            data=df, 
            hue='group',
            dodge=True,
            linewidth=1,
            legend=None
        )

        fig = g.get_figure()
        os.makedirs(f'{path}/figures', exist_ok=True)
        fig.savefig(f'{path}/figures/barplot_{name}.png')
        plt.close()

        for model in ['linsvc']:
            new_df = df_all[df_all["models"].str.contains(model)]
            sns.set(rc={'figure.figsize':(12, 8)})
            sns.set_style(style='whitegrid') 
            g = sns.barplot(
                x="batches",
                y='value',
                data=new_df,
                errorbar=None,
                hue='groups',
            )
            g = sns.stripplot(
                x="batches", 
                y='value', 
                data=new_df, 
                hue='groups',
                dodge=True,
                linewidth=1,
                legend=None
            )

            fig = g.get_figure()
            fig.savefig(f'{path}/figures/barplot_batches_{name}_{model}.png')
            plt.close()

            # get subframe of batch is b1
            g = sns.barplot(
                x="labels",
                y='value',
                data=df_all,
                errorbar=None,
                hue='groups',
            )
            g = sns.stripplot(
                x="labels", 
                y='value', 
                data=df_all, 
                hue='groups',
                dodge=True,
                linewidth=1,
                legend=None
            )

            fig = g.get_figure()
            plt.xticks(rotation = 45, ha='right', rotation_mode='anchor', fontsize=12)
            fig.savefig(f'{path}/figures/barplot_classes_{name}_{model}.png')
            plt.close()

def plot_individual_bars(batch_dates, args):
    args.groupkfold = 0  # gkf is always 0 for individual results
    args.ovr = 0
    concs = args.concs.split(',')
    cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"
    for name in ['mcc', 'acc']:
        df = {
            'value': [],
            'models': [],
            'group': [],
            'batches': [],
        }
        df_all = {
            'valid': {
                'labels': [],
                'classes': [],
                'models': [],
                'preds': [],
                'batches': [],
            },
            'test': {
                'labels': [],
                'classes': [],
                'models': [],
                'preds': [],
                'batches': [],
            }
        }
        for model in ['linsvc']:
            for date in batch_dates:
                date = date.split('-')[0]
                if args.output != '2models':
                    exp_name = f'{date}_binary{args.binary}_{args.n_features}' \
                                    f'_gkf{args.groupkfold}_ovr{args.ovr}_remblk1_{cropings}_{"_".join(concs)}'
                else:
                    exp_name = f'{date}_2models_{args.n_features}' \
                                    f'_gkf{args.groupkfold}_ovr{args.ovr}_{cropings}_{"_".join(concs)}'
                path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
                        f'{args.spd}spd/thr{args.threshold}/{args.train_on}/{exp_name}/{model}'
                results = json.load(open(f'{path}/saved_models/best_params_inputs_{model}_values.json'))
                batches = json.load(open(f'{path}/saved_models/unique_batches_inputs_{model}.json'))
                valid_scores = pd.read_csv(open(f'{path}/saved_models/{model}_valid_individual_results.csv'))
                test_scores = pd.read_csv(open(f'{path}/saved_models/{model}_test_individual_results.csv'))

                df['value'] += results[f'valid_{name}'] + results[f'test_{name}']
                df['models'] += [model] * len(results[f'valid_{name}']) + [model] * len(results[f'test_{name}'])
                df['group'] += ['valid'] * len(results[f'valid_{name}']) + ['test'] * len(results[f'test_{name}'])
                df['batches'] += [b[0] for b in batches['valid']] + [b[0] for b in batches['test']]
                df_all['valid']['classes'] += valid_scores.loc[:, 'classes'].to_list()
                df_all['test']['classes'] += test_scores.loc[:, 'classes'].to_list()
                df_all['valid']['labels'] += valid_scores.loc[:, 'labels'].to_list()
                df_all['test']['labels'] += test_scores.loc[:, 'labels'].to_list()
                df_all['valid']['batches'] += valid_scores.loc[:, 'batches'].to_list()
                df_all['test']['batches'] += test_scores.loc[:, 'batches'].to_list()
                df_all['valid']['preds'] += valid_scores.loc[:, 'preds'].to_list()
                df_all['test']['preds'] += test_scores.loc[:, 'preds'].to_list()
                df_all['valid']['models'] += [model] * len(valid_scores.loc[:, 'preds'].to_list())
                df_all['test']['models'] += [model] * len(test_scores.loc[:, 'preds'].to_list())

                df = pd.DataFrame(df)
                df_all2 = make_df_classes(df_all)
                for model in ['linsvc']:
                    new_df = df[df["models"].str.contains(model)]
                    sns.set(rc={'figure.figsize':(12, 8)})
                    sns.set_style(style='whitegrid') 
                    g = sns.barplot(
                        x="batches",
                        y='value',
                        data=new_df,
                        errorbar=None,
                        hue='group',
                    )
                    g = sns.stripplot(
                        x="batches", 
                        y='value', 
                        data=new_df, 
                        hue='group',
                        dodge=True,
                        linewidth=1,
                        legend=None
                    )

                    fig = g.get_figure()
                    fig.savefig(f'{path}/figures/barplot_indiv_batches_{date}_{name}_{model}.png')
                    plt.close()

                    # get subframe of batch is b1
                    g = sns.barplot(
                        x="labels",
                        y=name,
                        data=df_all2,
                        errorbar=None,
                        hue='groups',
                    )
                    g = sns.stripplot(
                        x="labels", 
                        y=name, 
                        data=df_all2, 
                        hue='groups',
                        dodge=True,
                        linewidth=1,
                        legend=None
                    )

                    fig = g.get_figure()
                    plt.xticks(rotation = 45, ha='right', rotation_mode='anchor', fontsize=12)
                    fig.savefig(f'{path}/figures/barplot_indiv_classes_{date}_{name}_{model}.png')
                    plt.close()


                    df = pd.DataFrame(df)
                    sns.set(rc={'figure.figsize':(12, 8)})
                    sns.set_style(style='whitegrid') 
                    g = sns.barplot(
                        x="batches",
                        y='value',
                        data=df,
                        errorbar=None,
                        hue='group',
                    )
                    g = sns.stripplot(
                        x="batches", 
                        y='value', 
                        data=df, 
                        hue='group',
                        dodge=True,
                        linewidth=1,
                        legend=None
                    )

                    fig = g.get_figure()
                    fig.savefig(f'{path}/figures/barplot_indiv_{date}_{name}_{model}_indiv.png')
                    plt.close()
                sns.set(rc={'figure.figsize':(12, 8)})
                sns.set_style(style='whitegrid') 
                g = sns.barplot(
                    x="models",
                    y='value',
                    data=df,
                    errorbar=None,
                    hue='group',
                )
                g = sns.stripplot(
                    x="models", 
                    y='value', 
                    data=df, 
                    hue='group',
                    dodge=True,
                    linewidth=1,
                    legend=None
                )

                fig = g.get_figure()
                fig.savefig(f'{path}/figures/barplot_indiv_{name}.png')
                plt.close()


if __name__ == '__main__':
    # Load data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--groupkfold', type=int, default=1)
    parser.add_argument('--n_features', type=int, default=-1)
    parser.add_argument('--binary', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='linsvc')
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
    args = parser.parse_args()

    batch_dates = [
        'B10-05-03-2024',
        'B1-02-02-2024', 'B2-02-21-2024', 'B3-02-29-2024', 
        'B4-03-01-2024', 'B5-03-13-2024', 'B6-03-29-2024',
        'B7-04-03-2024', "B8-04-15-2024", "B9-04-22-2024",
    ]
    plot_bars(batch_dates, args)
    plot_individual_bars(batch_dates, args)
    
