import os
from turtle import mode
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib import rcParams
# robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import r as robjects
from rpy2.robjects.conversion import localconverter
# sklearn
import sklearn
# from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from umap.umap_ import UMAP
from tabulate import tabulate

from msml.utils.pool_metrics import get_PCC, get_batches_euclidean, get_euclidean
from msml.utils.plotting import confidence_ellipse
from matplotlib.lines import Line2D
# from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy import stats
from statsmodels.stats.multitest import multipletests
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from torch.nn import functional as F
import seaborn as sns
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import matthews_corrcoef as MCC


def make_pval_table(data, unique_labels):
    print("Mann      pval min    n pvals < 0.05")
    table = pd.DataFrame(columns=['pval', 'n'])
    i = 0
    for i, label in enumerate(unique_labels[:-1]):
        for label2 in unique_labels[i+1:]:
            if label != label2 and label != 'pool' and label2 != 'pool':
                pvals = stats.mannwhitneyu(
                    data['inputs']['all'].iloc[np.argwhere(data['labels']['all'] == label).squeeze()], 
                    data['inputs']['all'].iloc[np.argwhere(data['labels']['all'] == label2).squeeze()]
                )
                tmp = multipletests(pvals[1], 0.05, 'fdr_bh')[1]
                table.loc[f'{label}_{label2}', 'pval'] = tmp.min()
                table.loc[f'{label}_{label2}', 'n'] = len([x for x in tmp if x < 0.05])
                i += 1
    print(tabulate(table))

    print('ttests')
    table = pd.DataFrame(columns=['pval'])
    i = 0
    for i, label in enumerate(unique_labels[:-1]):
        for label2 in unique_labels[i+1:]:
            if label != label2 and label != 'pool' and label2 != 'pool':
                pvals = stats.ttest_ind(
                    data['inputs']['all'].iloc[np.argwhere(data['labels']['all'] == label).squeeze()], 
                    data['inputs']['all'].iloc[np.argwhere(data['labels']['all'] == label2).squeeze()]
                )
                tmp = multipletests(pvals[1], 0.05, 'fdr_bh')[1]
                table.loc[f'{label}_{label2}', 'pval'] = tmp.min()
                table.loc[f'{label}_{label2}', 'n'] = len([x for x in tmp if x < 0.05])
                i += 1
    print(tabulate(table))


def log_fct(data, metric_name, metrics, make_pval=False):
    """
    Log metrics under a custom metric_name (e.g. 'inputs', 'bottleneck', etc.)
    """
    metrics = log_metrics(data, metrics, metric_name)
    return metrics


def log_ord(data, uniques, path, scaler_name, step_name, run=None):
    data_tmp = copy.deepcopy(data)
    data_tmp['inputs']['all'] = pd.concat((data['inputs']['all'], data['inputs']['urinespositives']), axis=0)
    data_tmp['labels']['all'] = np.concatenate((data['labels']['all'], data['labels']['urinespositives']))
    data_tmp['batches']['all'] = np.concatenate((data['batches']['all'], data['batches']['urinespositives']))
    data_tmp['batches_labels']['all'] = np.concatenate((data['batches_labels']['all'], data['batches_labels']['urinespositives']))
    data_tmp['manips']['all'] = np.concatenate((data['manips']['all'], data['manips']['urinespositives']))
    data_tmp['urines']['all'] = np.concatenate((data['urines']['all'], data['urines']['urinespositives']))
    data_tmp['concs']['all'] = np.concatenate((data['concs']['all'], data['concs']['urinespositives']))
    data_tmp['cats']['all'] = np.concatenate((data['cats']['all'], data['cats']['urinespositives']))
    data_tmp['names']['all'] = np.concatenate((data['names']['all'], data['names']['urinespositives']))
    data_tmp['urinespositives'] = {}
    data_tmp['urinespositives']['all'] = np.concatenate((
        np.array(['inoc' for _ in range(len(data['labels']['all']))]),
        np.array(['upos' for _ in range(len(data['labels']['urinespositives']))])
    ))

    unique_labels = uniques['labels']
    unique_batches = uniques['batches']
    unique_manips = uniques['manips']
    unique_urines = uniques['urines']
    unique_concs = uniques['concs']
    unique_urinespositives = np.unique(data_tmp['urinespositives']['all'])
    log_ORD({'model': PCA(n_components=2), 'name': f'PCA_{scaler_name}'}, data_tmp,
            {'batches_labels': unique_batches, 'labels': unique_labels,
             'manips': unique_manips, 'urines': unique_urines, 'concs': unique_concs,
             'urinespositives': unique_urinespositives
             }, path, scaler_name, step_name, run)
    log_ORD({'model': UMAP(n_components=2), 'name': f'UMAP_{scaler_name}'}, data_tmp,
            {'batches_labels': unique_batches, 'labels': unique_labels,
             'manips': unique_manips, 'urines': unique_urines, 'concs': unique_concs,
             'urinespositives': unique_urinespositives
             }, path, scaler_name, step_name, run)
    if unique_labels is not None:
        try:
            log_LDA(LDA, data_tmp, {'batches_labels': unique_batches, 'labels': unique_labels}, path, scaler_name)
        except Exception as e:
            print(e)
            print('\n\nLDA failed\n\n')


def pyGPCA(data, group, name, metrics):
    gPCA = importr('gPCA')

    newdata = robjects.r.matrix(robjects.FloatVector(np.array(data['inputs'][group]).reshape(-1)), nrow=data['inputs'][group].shape[0])
    new_batches = robjects.r.matrix(robjects.IntVector(data['batches'][group]), nrow=data['inputs'][group].shape[0])

    results = gPCA.gPCA_batchdetect(newdata, new_batches)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results = {key : np.array(robjects.conversion.rpy2py(results.rx2(key))) for key in results.names }

    if 'pool' in name:
        delta_name = 'delta_pool'
        name = name.split('_')[1]
    else:
        delta_name = 'delta'
    if name not in metrics:
        metrics[name] = {}
    metrics[name][delta_name] = results['delta'][0]
    # metrics[name]['delta.pval'] = results[1]
    return metrics, results


def log_ORD(ordin, data, uniques, path, scaler_name, step_name='inputs', run=None):
    model = ordin['model']
    data = copy.deepcopy(data)
    pcs_train = model.fit_transform(data['inputs']['all'])
    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['PC 1', 'PC 2'])
    if hasattr(model, 'explained_variance_ratio_'):
        ev = model.explained_variance_ratio_
        pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
        pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
    else:
        pc1 = 'Component_1'
        pc2 = 'Component_2'

    for name in list(uniques.keys()):
        if uniques[name] is None:
            continue
        data_name = name
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_xlabel(pc1, fontsize=20)
        ax.set_ylabel(pc2, fontsize=20)

        # num_targets = len(uniques[name])
        cmap = plt.cm.tab20

        cols = cmap(np.linspace(0, 1, len(uniques[name]) + 1))
        # colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
        colors_list = {name: [] for name in ['all']}
        data1_list = {name: [] for name in ['all']}
        data2_list = {name: [] for name in ['all']}
        new_labels = {name: [] for name in ['all']}
        new_cats = {name: [] for name in ['all']}

        ellipses = []
        unique_cats_train = np.array([])
        for df_name, df, labels in zip(['all'], [pcs_train_df], [data[data_name]['all']]):
            for t, target in enumerate(uniques[name]):
                if name == 'batches':
                    target = np.argwhere(target == uniques[name])[0][0]
                indices_to_keep = [True if x == target else False for x in list(labels)]
                data1 = list(df.loc[indices_to_keep, 'PC 1'])
                new_labels[df_name] += [target for _ in range(len(data1))]
                new_cats[df_name] += [target for _ in range(len(data1))]

                data2 = list(df.loc[indices_to_keep, 'PC 2'])
                data1_list[df_name] += [data1]
                data2_list[df_name] += [data2]
                colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
                if (df_name == 'all' or target not in unique_cats_train) and sum(indices_to_keep) > 1:
                    unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                    try:
                        confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                                 edgecolor=cols[t],
                                                                 train_set=True)
                        ellipses += [confidence_ellipses[1]]
                    except Exception as e:
                        print(t, target, name, 'Whoopsie', e)
                        pass

        for df_name, marker in zip(list(data1_list.keys()), ['o']):
            data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
            colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
            data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
            data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
            data2 = data_colors_vector[:, 1]
            col = data_colors_vector[:, 2:]
            data1 = data_colors_vector[:, 0]

            ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
            custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(uniques[name]) + 1)]
            if 'pool' in uniques[name]:
                uniques[name][np.argwhere(uniques[name] == 'pool')] = 'QC'

            if len(custom_lines) < 30:
                ax.legend(custom_lines, uniques[name].tolist(), fontsize=10)

        plt.show()
        plt.savefig(f'{path}/{ordin["name"]}_{name}_{scaler_name}.png')
        if run is not None:
            run[f'{step_name}/{ordin["name"]}/{name}'].upload(fig)
        plt.close()


def log_LDA(ordin, data, uniques, path, scaler_name, model_name="LDA"):
    for name in list(uniques.keys()):
        if len(uniques[name]) == 1:
            continue
        train_nums = np.arange(0, len(data['labels']['all']))
        scores = []
        # Remove samples from unwanted batches
        if len(uniques['batches']) == 1:
            skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
            splitter = skf.split(train_nums, data['labels']['all'])
        else:
            skf = sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=False, random_state=None)
            splitter = skf.split(train_nums, data['labels']['all'], data['batches']['all'])
        for i, (train_inds, valid_inds) in enumerate(splitter):
            if len(uniques[name]) > 2:
                n_comp = 2
            else:
                n_comp = 1
            model = ordin(n_components=n_comp)

            pcs_train = model.fit_transform(data['inputs']['all'].iloc[train_inds], data[name]['all'][train_inds])
            pcs_valid = model.transform(data['inputs']['all'].iloc[valid_inds])
            scores += [model.score(data['inputs']['all'].iloc[valid_inds], data[name]['all'][valid_inds])]
            if pcs_train.shape[1] == 1:
                n_comp = 1
            if i == 0:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111)

                if n_comp > 1:
                    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['LD1', 'LD2'])
                    pcs_valid_df = pd.DataFrame(data=pcs_valid, columns=['LD1', 'LD2'])
                    # pcs_test_df = pd.DataFrame(data=pcs_test, columns=['LD1', 'LD2'])
                else:
                    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['LD1'])
                    pcs_valid_df = pd.DataFrame(data=pcs_valid, columns=['LD1'])
                    # pcs_test_df = pd.DataFrame(data=pcs_test, columns=['LD1'])

                if hasattr(model, 'explained_variance_ratio_'):
                    ev = model.explained_variance_ratio_
                    pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
                    pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
                else:
                    pc1 = 'Component_1'
                    pc2 = 'Component_2'

                ax.set_xlabel(pc1, fontsize=15)
                ax.set_ylabel(pc2, fontsize=15)
                ax.set_title("2 component LDA", fontsize=20)

                # num_targets = len(uniques[name])
                cmap = plt.cm.tab20

                cols = cmap(np.linspace(0, 1, len(uniques[name]) + 1))
                # colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
                colors_list = {name: [] for name in ['train_data', 'valid_data']}
                data1_list = {name: [] for name in ['train_data', 'valid_data']}
                data2_list = {name: [] for name in ['train_data', 'valid_data']}
                new_labels = {name: [] for name in ['train_data', 'valid_data']}
                new_cats = {name: [] for name in ['train_data', 'valid_data']}

                ellipses = []
                unique_cats_train = np.array([])

                for df_name, df, labels in zip(['train_data', 'valid_data'],
                                               [pcs_train_df, pcs_valid_df],
                                               [data[name]['all'][train_inds], data[name]['all'][valid_inds]]):
                    for t, target in enumerate(uniques[name]):
                        indices_to_keep = [True if x == target else False for x in
                                           list(labels)]  # 0 is the name of the column with target values
                        data1 = list(df.loc[indices_to_keep, 'LD1'])
                        new_labels[df_name] += [target for _ in range(len(data1))]
                        new_cats[df_name] += [target for _ in range(len(data1))]

                        data1_list[df_name] += [data1]
                        colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
                        if n_comp > 1:
                            data2 = list(df.loc[indices_to_keep, 'LD2'])
                            data2_list[df_name] += [data2]
                        if sum(indices_to_keep) > 1 and df_name == 'all_data' or target not in unique_cats_train:
                            unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                            if n_comp > 1:
                                try:
                                    confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                                            edgecolor=cols[t],
                                                                            train_set=True)
                                    ellipses += [confidence_ellipses[1]]
                                except Exception as e:
                                    print(e)
                                    pass

                for df_name, marker in zip(list(data1_list.keys()), ['o', '*']):
                    data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
                    colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
                    if n_comp > 1:
                        data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
                        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
                        data1 = data_colors_vector[:, 0]
                        data2 = data_colors_vector[:, 1]
                        col = data_colors_vector[:, 2:]
                        ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
                    else:
                        data_colors_vector = np.concatenate((data1_vector, colors_vector), axis=1)
                        data1 = data_colors_vector[:, 0]
                        col = data_colors_vector[:, 1:]
                        ax.scatter(data1, np.random.random(len(data1)), s=50, alpha=1.0, c=col, label=new_labels[df_name],
                                   marker=marker)

                    custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(uniques[name]) + 1)]
                    ax.legend(custom_lines, uniques[name].tolist())

                plt.show()
                plt.savefig(f'{path}/{model_name}_{name}_{scaler_name}.png')
                plt.close()


def log_CCA(ordin, data, unique_cats, epoch):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    model = ordin['model']

    # try:
    #     train_cats = OneHotEncoder().fit_transform(np.stack([np.argwhere(unique_cats == x) for x in data['labels']['train']]).reshape(-1, 1)).toarray()
    # except Exception as e:
    #     print(e)
    #     pass
    # test_cats = [np.argwhere(unique_cats == x) for x in test_labels]
    # inference_cats = [np.argwhere(unique_cats == x) for x in inference_labels]

    pcs_train, _ = model.fit_transform(data['inputs']['train'], data['cats']['train'])
    pcs_test = model.transform(data['inputs']['valid'])
    pcs_inference = model.transform(data['inputs']['test'])

    pcs_train_df = pd.DataFrame(data=pcs_train, columns=['PC 1', 'PC 2'])
    pcs_test_df = pd.DataFrame(data=pcs_test, columns=['PC 1', 'PC 2'])
    pcs_inference_df = pd.DataFrame(data=pcs_inference, columns=['PC 1', 'PC 2'])
    if hasattr(model, 'explained_variance_ratio_'):
        ev = model.explained_variance_ratio_
        pc1 = 'Component_1 : ' + str(np.round(ev[0] * 100, decimals=2)) + "%"
        pc2 = 'Component_2 : ' + str(np.round(ev[1] * 100, decimals=2)) + "%"
    else:
        pc1 = 'Component_1'
        pc2 = 'Component_2'

    ax.set_xlabel(pc1, fontsize=15)
    ax.set_ylabel(pc2, fontsize=15)
    ax.set_title(f"2 component {ordin['name']}", fontsize=20)

    # num_targets = len(unique_cats)
    cmap = plt.cm.tab20

    cols = cmap(np.linspace(0, 1, len(unique_cats) + 1))
    # colors = rcParams['axes.prop_cycle'] = cycler(color=cols)
    colors_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
    data1_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
    data2_list = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
    new_labels = {name: [] for name in ['train_data', 'valid_data', 'test_data']}
    new_cats = {name: [] for name in ['train_data', 'valid_data', 'test_data']}

    ellipses = []
    unique_cats_train = np.array([])
    for df_name, df, labels in zip(['train_data', 'valid_data', 'test_data'],
                                   [pcs_train_df, pcs_test_df, pcs_inference_df],
                                   [data['labels']['train'], data['labels']['valid'], data['labels']['test']]):
        for t, target in enumerate(unique_cats):
            # final_labels = list(train_labels)
            indices_to_keep = [True if x == target else False for x in
                               list(labels)]  # 0 is the name of the column with target values
            data1 = list(df.loc[indices_to_keep, 'PC 1'])
            new_labels[df_name] += [target for _ in range(len(data1))]
            new_cats[df_name] += [target for _ in range(len(data1))]

            data2 = list(df.loc[indices_to_keep, 'PC 2'])
            data1_list[df_name] += [data1]
            data2_list[df_name] += [data2]
            colors_list[df_name] += [np.array([[cols[t]] * len(data1)])]
            if sum(indices_to_keep) > 1 and df_name == 'train_data' or target not in unique_cats_train:
                unique_cats_train = np.unique(np.concatenate((new_labels[df_name], unique_cats_train)))
                try:
                    confidence_ellipses = confidence_ellipse(np.array(data1), np.array(data2), ax, 1.5,
                                                             edgecolor=cols[t],
                                                             train_set=True)
                    ellipses += [confidence_ellipses[1]]
                except:
                    pass

    for df_name, marker in zip(list(data1_list.keys()), ['o', 'x', '*']):
        data1_vector = np.hstack([d for d in data1_list[df_name] if len(d) > 0]).reshape(-1, 1)
        colors_vector = np.hstack([d for d in colors_list[df_name] if d.shape[1] > 0]).squeeze()
        data2_vector = np.hstack(data2_list[df_name]).reshape(-1, 1)
        data_colors_vector = np.concatenate((data1_vector, data2_vector, colors_vector), axis=1)
        data2 = data_colors_vector[:, 1]
        col = data_colors_vector[:, 2:]
        data1 = data_colors_vector[:, 0]

        ax.scatter(data1, data2, s=50, alpha=1.0, c=col, label=new_labels[df_name], marker=marker)
        custom_lines = [Line2D([0], [0], color=cmap(x), lw=4) for x in np.linspace(0, 1, len(unique_cats) + 1)]
        if len(custom_lines) < 30:
            ax.legend(custom_lines, unique_cats.tolist())

    fig.savefig(f'{ordin["name"]}.png')
    plt.close()


def log_pool_metrics(data, batches, metrics, form):
    metric = {}

    for group in ['all']:
        if 'pool' not in group:
            try:
                data[group] = data[group].to_numpy()
                # data[f'{group}_pool'] = data[f'{group}_pool'].to_numpy()
            except Exception as e:
                print(e)

            metric[group] = {}
            batch_train_samples = [[i for i, batch in enumerate(batches[group].tolist()) if batch == b] for b in
                                   np.unique(batches[group])]

            batches_sample_indices = {
                group: batch_train_samples,
                # f'{group}_pool': batch_pool_samples,
            }
            # Average Pearson's Correlation Coefficients
            try:
                metric = get_PCC(data, batches, group, metric)
            except Exception as e:
                print(e)

            # Batch avg distance
            try:
                metric = get_batches_euclidean(data, batches_sample_indices, cdist, group, metric)
            except Exception as e:
                print(e)

            # avg distance
            try:
                metric = get_euclidean(data, pdist, group, metric)
            except Exception as e:
                print(e)

        # metrics[f'pool_metrics_{form}'] = metric
    if form not in metrics:
        metrics[form] = {}
    for m in metric:
        metrics[form][m] = metric[m]

    return metrics


def batch_entropy(proba):
    prob_list = []
    for prob in proba:
        loc = 0
        for p in prob:
            loc -= p * np.log(p + 1e-8)
        prob_list += [loc]
    return np.mean(prob_list)


def get_metrics(data, metrics, form):
    # sets are grouped togheter for a single metric
    new_data = data
    blancs_inds = np.argwhere(data['labels']['all'] == 'blanc').flatten()
    new_data['labels']['all'] = new_data['labels']['all'][blancs_inds]
    new_data['cats']['all'] = new_data['cats']['all'][blancs_inds]
    new_data['concs']['all'] = new_data['concs']['all'][blancs_inds]
    new_data['urines']['all'] = new_data['urines']['all'][blancs_inds]
    new_data['manips']['all'] = new_data['manips']['all'][blancs_inds]
    new_data['batches']['all'] = new_data['batches']['all'][blancs_inds]
    # maintain alignment for batches_labels if present
    if 'batches_labels' in new_data and 'all' in new_data['batches_labels']:
        new_data['batches_labels']['all'] = new_data['batches_labels']['all'][blancs_inds]
    # choose which batch key to use for domain (batch) metrics
    batch_domain_key = 'batches_labels' if 'batches_labels' in new_data else 'batches'
    new_data['inputs']['all'] = new_data['inputs']['all'].iloc[blancs_inds]
    knns = {repres: KNeighborsClassifier(n_neighbors=20) for repres in ['domains', 'labels']}

    if form not in metrics:
        metrics[form] = {}
    for group in ['all']:
        # metrics[group] = {m : {'labels': [], 'domains': []} for m in ['lisi', 'kbet', 'silhouette',
        # 'adjusted_rand_score', 'adjusted_mutual_info_score']}
        if group not in metrics[form]:
            metrics[form][group] = {}
        # keep only data where labels == blanc
        if group == 'all':
            knns['domains'].fit(new_data['inputs'][group], new_data[batch_domain_key][group])
            knns['labels'].fit(new_data['inputs'][group], new_data['cats'][group])
        if 'pool' not in group or 'all_pool' == group:
            # for metric, funct in zip(['lisi', 'silhouette', 'kbet'], [rLISI, silhouette_score, rKBET]):
            for metric, funct in zip(['silhouette'], [silhouette_score]):
                try:
                    metrics[form][group][metric] = {'labels': None, 'domains': None}
                    metrics[form][group][metric]['domains'] = \
                        funct(new_data['inputs'][group], new_data[batch_domain_key][group])
                    if 'pool' not in group:
                        metrics[form][group][metric]['labels'] =\
                            funct(new_data['inputs'][group], new_data['cats'][group])
                except Exception as e:
                    print(e)

            domain_preds = knns['domains'].predict(new_data['inputs'][group].values)
            metrics[form][group]['shannon'] = {'labels': None, 'domains': None}
            metrics[form][group]['shannon']['domains'] =\
                batch_entropy(knns['domains'].predict_proba(new_data['inputs'][group].values))
            if 'pool' not in group:
                labels_preds = knns['labels'].predict(new_data['inputs'][group].values)
                metrics[form][group]['shannon']['labels'] =\
                    batch_entropy(knns['labels'].predict_proba(new_data['inputs'][group].values))

            for metric, funct in zip(
                    ['adjusted_rand_score', 'adjusted_mutual_info_score'],
                    [adjusted_rand_score, adjusted_mutual_info_score]):
                metrics[form][group][metric] = {'labels': None, 'domains': None}
                metrics[form][group][metric]['domains'] = funct(new_data[batch_domain_key][group], domain_preds)
                if 'pool' not in group:
                    metrics[form][group][metric]['labels'] = funct(new_data[batch_domain_key][group], labels_preds)

    return metrics


def batch_f1_score(batch_score, class_score):
    return 2 * (1 - batch_score) * (class_score) / (1 - batch_score + class_score)


def log_metrics(data, metrics, metric_name):
    """
    Log metrics under a custom metric_name (e.g. 'inputs', 'bottleneck', etc.)
    """
    if metric_name not in metrics:
        metrics[metric_name] = {}
    metrics = get_metrics(data, metrics, metric_name)
    for repres in ['inputs']:
        for metric in ['adjusted_rand_score', 'adjusted_mutual_info_score']:
            for group in ['all']:
                try:
                    metrics[metric_name][group][metric]['F1'] = batch_f1_score(
                        batch_score=metrics[metric_name][group][metric]['domains'],
                        class_score=metrics[metric_name][group][metric]['labels'],
                    )
                except Exception as e:
                    print(e)
                    metrics[metric_name][group][metric]['F1'] = -1
    return metrics


def save_confusion_matrix(fig, name, acc, mcc, group):
    # sns_plot = sns.heatmap(df, annot=True, square=True, cmap="YlGnBu",
    #                        annot_kws={"size": 35 / np.sqrt(len(df))})
    # fig = sns_plot.get_figure()
    dirs = '/'.join(name.split('/')[:-1])
    name = name.split('/')[-1] + '_1batch'
    plt.title(f'Confusion Matrix (acc={np.round(np.mean(acc), 2)} '
              f'+- {np.round(np.std(acc), 2)}, '
              f'mcc={np.round(np.mean(mcc), 2)} '
              f'+- {np.round(np.std(mcc), 2)})')
    os.makedirs(f'{dirs}/', exist_ok=True)
    stuck = True
    while stuck:
        try:
            fig.savefig(f"{dirs}/cm_{name}_{group}.png")
            stuck = False
        except Exception as e:
            print(e)
            print('stuck...')
    plt.close()


def log_neptune(run, traces, best_scores):
    '''
    Log the scores to neptune
    '''
    for g in ['train', 'valid', 'test']:
        if isinstance(traces['acc'][g], list):
            if len(traces['acc'][g]) == 0:
                continue
        elif isinstance(traces['acc'][g], (float, np.float64)) is False:
            continue
        run[f'{g}/acc'].log(traces['acc'][g])
        run[f'{g}/mcc'].log(traces['mcc'][g])
        try:
            run[f'{g}/bact_acc'].log(traces['bact_acc'][g])
            run[f'{g}/bact_mcc'].log(traces['bact_mcc'][g])
        except Exception as e:
            print(e)
        try:
            run[f'{g}/ari'].log(best_scores['ari'][g])
            run[f'{g}/ami'].log(best_scores['ami'][g])
            run[f'{g}/nBE'].log(best_scores['nbe'][g])
        except Exception as e:
            print(e)
    if 'posurines' in traces['acc']:
        try:
            run['posurines/acc'].log(traces['acc']['posurines'])
            run['posurines/mcc'].log(traces['mcc']['posurines'])
        except Exception as e:
            print(e)


# DVC logging function: logs metrics and artifacts to dvclive (DVC)
def log_dvclive(live, traces=None, best_scores=None, artifacts=None):
    """
    Log metrics and artifacts to DVC/dvclive.
    Args:
        traces (dict): Dictionary of metrics to log (e.g., per-epoch or per-step metrics).
        best_scores (dict): Dictionary of best scores/metrics to log.
        artifacts (list): List of file paths to log as artifacts (plots, models, etc.).
    """
    if best_scores is not None:
        for key, value in best_scores.items():
            live.log_metric(key, value)
    if artifacts is not None:
        for path in artifacts:
            live.log_artifact(path)
    for i in range(len(traces['mcc']['train'])):
        if traces is not None:
            for key in ['acc', 'mcc']:
                for g in ['train', 'valid', 'test']:
                    if key in traces and g in traces[key]:
                        if isinstance(traces[key][g], list):
                            if len(traces[key][g]) == 0:
                                continue
                        elif isinstance(traces[key][g], (float, np.float64)) is False:
                            continue
                        live.log_metric(f'{key}_{g}', traces[key][g][i])
        live.next_step()


# Calculate tpr by bacterium
def calculate_tpr_per_bact(df, bacteria_cols):
    true_positives = {}
    true_positivess = {b: [] for b in bacteria_cols}
    true_positivess_cv = {b: [] for b in bacteria_cols}
    for bacterium in bacteria_cols:
        positives = df[df['labels'] == df['preds']][df['labels'] == bacterium].shape[0]
        total_predictions = (df['labels'] == bacterium).sum()
        true_positives[bacterium] = positives / total_predictions
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    for bacterium in bacteria_cols:
        for batch in df[batch_col].unique():
            tmp = df.copy()
            tmp = tmp[tmp[batch_col] == batch]
            if bacterium in tmp['labels'].unique():
                true_positivess[bacterium] += [
                    tmp[tmp['labels'] == tmp['preds']][tmp['labels'] == bacterium].shape[0]
                    / tmp[tmp['labels'] == bacterium].shape[0]
                ]
    for bacterium in bacteria_cols:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            if bacterium in tmp['labels'].unique():
                try:
                    true_positivess_cv[bacterium] += [
                        tmp[
                            tmp['labels'] == tmp['preds']
                        ][tmp['labels'] == bacterium].shape[0] / tmp[tmp['labels'] == bacterium].shape[0]]
                except Exception as e:
                    print(e)
    return true_positives, true_positivess, true_positivess_cv


# Calculate tpr by batch
def calculate_tpr_per_batch(df, batches):
    # Use unified batches_labels column if available
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    true_positives = {}
    true_positivess = {b: [] for b in batches}
    true_positivess_cv = {b: [] for b in batches}
    for batch in batches:
        corrects = df[df['labels'] == df['preds']][df[batch_col] == batch].shape[0]
        total = (df[batch_col] == batch).sum()
        true_positives[batch] = corrects / total
    for batch in batches:
        # for bacterium in df['labels'].unique():
        tmp = df.copy()
        tmp = tmp[tmp[batch_col] == batch]
        # if bacterium in tmp['labels'].unique():
        true_positivess[batch] += [tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp.shape[0]]
    for batch in batches:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            # if bacterium in tmp['labels'].unique():
            try:
                true_positivess_cv[batch] += [tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp.shape[0]]
            except Exception as e:
                print(e)
    return true_positives, true_positivess, true_positivess_cv


# Calculate tnr by bacterium
def calculate_tnr_per_bact(df, bacteria_cols):
    true_negatives = {}
    true_negativess = {b: [] for b in bacteria_cols}
    true_negativess_cv = {b: [] for b in bacteria_cols}
    for bacterium in bacteria_cols:
        positives = df[df['labels'] == df['preds']][df['labels'] == bacterium].shape[0]
        total_predictions = (df['labels'] == bacterium).sum()
        true_negatives[bacterium] = positives / total_predictions
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    for bacterium in bacteria_cols:
        for batch in df[batch_col].unique():
            tmp = df.copy()
            tmp = tmp[tmp[batch_col] == batch]
            if bacterium in tmp['labels'].unique():
                true_negativess[bacterium] += [
                    tmp[tmp['labels'] == tmp['preds']][tmp['labels'] == bacterium].shape[0]
                    / tmp[tmp['labels'] == bacterium].shape[0]
                ]
    for bacterium in bacteria_cols:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            if bacterium in tmp['labels'].unique():
                try:
                    true_negativess_cv[bacterium] += [tmp[tmp['labels'] == tmp['preds']][
                        tmp['labels'] == bacterium
                    ].shape[0] / tmp[tmp['labels'] == bacterium].shape[0]]
                except Exception as e:
                    print(e)
    return true_negatives, true_negativess, true_negativess_cv


# Calculate tnr by batch
def calculate_tnr_per_batch(df, batches):
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    true_negatives = {}
    true_negativess = {b: [] for b in batches}
    true_negativess_cv = {b: [] for b in batches}
    for batch in batches:
        corrects = df[df['labels'] == df['preds']][df[batch_col] == batch].shape[0]
        total = (df[batch_col] == batch).sum()
        true_negatives[batch] = corrects / total
    for batch in batches:
        # for bacterium in df['labels'].unique():
        tmp = df.copy()
        tmp = tmp[tmp[batch_col] == batch]
        # if bacterium in tmp['labels'].unique():
        true_negativess[batch] += [tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp.shape[0]]
    for batch in batches:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            # if bacterium in tmp['labels'].unique():
            try:
                true_negativess_cv[batch] += [tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp.shape[0]]
            except Exception as e:
                print(e)
    return true_negatives, true_negativess, true_negativess_cv


def calculate_acc_per_bact(df, bacteria_cols):
    acc = {}
    accs = {b: [] for b in bacteria_cols}
    accs_cv = {b: [] for b in bacteria_cols}
    for bacterium in bacteria_cols:
        tmp = df.copy()
        tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        acc[bacterium] = ACC(tmp['labels'], tmp['preds'])
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    for bacterium in bacteria_cols:
        for batch in df[batch_col].unique():
            tmp = df.copy()
            tmp = tmp[tmp[batch_col] == batch]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                accs[bacterium] += [ACC(tmp['labels'], tmp['preds'])]
    for bacterium in bacteria_cols:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                accs_cv[bacterium] += [ACC(tmp['labels'], tmp['preds'])]

    return acc, accs, accs_cv


def calculate_acc_per_batch(df, batches):
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    mcc = {}
    mccs = {b: [] for b in batches}
    mccs_cv = {b: [] for b in batches}
    for batch in batches:
        tmp = df.copy().iloc[df[df[batch_col] == batch].index]
        mcc[batch] = ACC(tmp['labels'], tmp['preds'])
    for batch in batches:
        # for bacterium in df['labels'].unique():
        tmp = df.copy()
        tmp = tmp[tmp[batch_col] == batch]
        # if bacterium in tmp['labels'].unique():
        # tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        # tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        mccs[batch] += [ACC(tmp['labels'], tmp['preds'])]
    for batch in batches:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            # if bacterium in tmp['labels'].unique():
            # tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
            # tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
            mccs_cv[batch] += [ACC(tmp['labels'], tmp['preds'])]
    return mcc, mccs, mccs_cv


def calculate_precision_per_bact(df, bacteria_cols):
    precision = {}
    precisions = {b: [] for b in bacteria_cols}
    precisions_cv = {b: [] for b in bacteria_cols}
    for bacterium in bacteria_cols:
        if bacterium not in df['labels'].unique():
            # precision[bacterium] = 0
            # precisions[bacterium] = []
            # precisions_cv[bacterium] = []
            continue    
        tmp = df.copy()
        tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        corrects = df[df['labels'] == df['preds']][df['labels'] == bacterium].shape[0]
        total = df[df['labels'] == bacterium].shape[0]
        precision[bacterium] = corrects / total
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    for bacterium in bacteria_cols:
        for batch in df[batch_col].unique():
            tmp = df.copy()
            tmp = tmp[tmp[batch_col] == batch]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                try:
                    precisions[bacterium] += [
                        tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp[tmp['preds'] == 1].shape[0]
                    ]
                except Exception as e:
                    print(e)
    for bacterium in bacteria_cols:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                try:
                    precisions_cv[bacterium] += [
                        tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp[tmp['preds'] == 1].shape[0]
                    ]
                except Exception as e:
                    print(e)
    return precision, precisions, precisions_cv


def calculate_precision_per_batch(df, batches):
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    precision = {}
    precisions = {b: [] for b in batches}
    precisions_cv = {b: [] for b in batches}
    for batch in batches:
        tmp = df.copy().iloc[df[df[batch_col] == batch].index]
        corrects = tmp[tmp['labels'] == tmp['preds']].shape[0]
        total = tmp.shape[0]
        precision[batch] = corrects / total
    for batch in batches:
        # for bacterium in df['labels'].unique():
        tmp = df.copy()
        tmp = tmp[tmp[batch_col] == batch]
        # if bacterium in tmp['labels'].unique():
        # tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        # tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        try:
            precisions[batch] += [tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp[tmp['preds'] == 1].shape[0]]
        except Exception as e:
            print(e)
    for batch in batches:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            # if bacterium in tmp['labels'].unique():
            # tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
            # tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
            try:
                precisions_cv[batch] += [tmp[tmp['labels'] == tmp['preds']].shape[0] / tmp[tmp['preds'] == 1].shape[0]]
            except Exception as e:
                print(e)
    return precision, precisions, precisions_cv


def calculate_mcc_per_bact_old(df, bacteria_cols):
    mcc = {}
    mccs = {b: [] for b in bacteria_cols}
    mccs_cv = {b: [] for b in bacteria_cols}
    for bacterium in bacteria_cols:
        tmp = df.copy()
        tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        mcc[bacterium] = MCC(tmp['labels'], tmp['preds'])
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    for bacterium in bacteria_cols:
        for batch in df[batch_col].unique():
            tmp = df.copy()
            tmp = tmp[tmp[batch_col] == batch]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                mccs[bacterium] += [MCC(tmp['labels'], tmp['preds'])]
    for bacterium in bacteria_cols:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                mccs_cv[bacterium] += [MCC(tmp['labels'], tmp['preds'])]
    return mcc, mccs, mccs_cv


def calculate_mcc_per_bact(df, bacteria_cols):
    mcc = {}
    mccs = {b: [] for b in bacteria_cols}
    mccs_cv = {b: [] for b in bacteria_cols}
    for bacterium in bacteria_cols:
        tmp = df.copy()
        tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
        tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
        mcc[bacterium] = MCC(tmp['labels'], tmp['preds'])
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    for bacterium in bacteria_cols:
        for batch in df[batch_col].unique():
            tmp = df.copy()
            tmp = tmp[tmp[batch_col] == batch]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                mccs[bacterium] += [MCC(tmp['labels'], tmp['preds'])]
    for bacterium in bacteria_cols:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            if bacterium in tmp['labels'].unique():
                tmp['labels'] = tmp['labels'].apply(lambda x: 1 if x == bacterium else 0)
                tmp['preds'] = tmp['preds'].apply(lambda x: 1 if x == bacterium else 0)
                mccs_cv[bacterium] += [MCC(tmp['labels'], tmp['preds'])]
    return mcc, mccs, mccs_cv


def calculate_mcc_per_batch(df, batches):
    batch_col = 'batches_labels' if 'batches_labels' in df.columns else 'batches'
    mcc = {}
    mccs = {b: [] for b in batches}
    mccs_cv = {b: [] for b in batches}
    for batch in batches:
        tmp = df.copy().iloc[df[df[batch_col] == batch].index]
        mcc[batch] = MCC(tmp['labels'], tmp['preds'])
    for batch in batches:
        # for bacterium in df['labels'].unique():
        tmp = df.copy()
        tmp = tmp[tmp[batch_col] == batch]
        # if bacterium in tmp['labels'].unique():
        # tmp['labels'] = tmp['labels']#.apply(lambda x: 1 if x == bacterium else 0)
        # tmp['preds'] = tmp['preds']#.apply(lambda x: 1 if x == bacterium else 0)
        mccs[batch] += [MCC(tmp['labels'], tmp['preds'])]
    for batch in batches:
        for cv in df['cv'].unique():
            tmp = df.copy()
            tmp = tmp[tmp['cv'] == cv]
            # if bacterium in tmp['labels'].unique():
            # tmp['labels'] = tmp['labels']#.apply(lambda x: 1 if x == bacterium else 0)
            # tmp['preds'] = tmp['preds']#.apply(lambda x: 1 if x == bacterium else 0)
            mccs_cv[batch] += [MCC(tmp['labels'], tmp['preds'])]
    return mcc, mccs, mccs_cv


def make_graph(metrics, metric, category, cols, title, path, run, mlops='neptune'):
    # Prepare data for plotting
    metric_df = pd.DataFrame({
        category: cols,
        'Valid': [metrics['valid'][b] for b in cols],
        'Test': [metrics['test'][b] for b in cols]
    })

    metric_melted_df = metric_df.melt(id_vars=category, var_name='Group', value_name=metric)

    # Plot metric by bacterium
    plt.figure(figsize=(14, 8))
    sns.barplot(x=category, y=metric, hue='Group', data=metric_melted_df)
    for i, bacterium in enumerate(cols):
        # Add errorbars for each bacterium

        plt.errorbar(
            i - 0.2, metrics['valid'][bacterium],
            yerr=([0], [np.std(metrics['valids'][bacterium])]),
            fmt='', color='black', capsize=5
        )
        plt.errorbar(
            i + 0.2, metrics['test'][bacterium],
            yerr=([0], [np.std(metrics['tests'][bacterium])]),
            fmt='', color='black', capsize=5
        )
        # Add a scatter plot for each bacterium
        plt.scatter(
            [i - 0.2 + np.random.uniform(-0.02, 0.02) for _ in range(len(metrics['valids'][bacterium]))],
            metrics['valids'][bacterium], color='black', s=(rcParams['lines.markersize'] ** 2)/3
        )
        plt.scatter(
            [i + 0.2 + np.random.uniform(-0.02, 0.02) for _ in range(len(metrics['tests'][bacterium]))],
            metrics['tests'][bacterium], color='black', s=(rcParams['lines.markersize'] ** 2)/3
        )
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{path}.png')
    plt.close()
    if mlops == 'neptune':
        run[f'{category}/{metric}'].upload(f'{path}.png')
    elif mlops == 'dvclive':
        run.log_artifact(f'{path}.png')
    # Plot metric by bacterium
    plt.figure(figsize=(14, 8))
    sns.barplot(x=category, y=metric, hue='Group', data=metric_melted_df)
    for i, bacterium in enumerate(cols):
        # Add errorbars for each bacterium
        plt.errorbar(
            i - 0.2, metrics['valid'][bacterium],
            yerr=([0], [np.std(metrics['valid_cv'][bacterium])]),
            fmt='', color='black', capsize=5
        )
        plt.errorbar(
            i + 0.2, metrics['test'][bacterium],
            yerr=([0], [np.std(metrics['test_cv'][bacterium])]),
            fmt='', color='black', capsize=5
        )
        # Add a scatter plot for each bacterium
        plt.scatter([
            i - 0.2 + np.random.uniform(-0.02, 0.02) for _ in range(len(metrics['valid_cv'][bacterium]))],
            metrics['valid_cv'][bacterium], color='black', s=(rcParams['lines.markersize'] ** 2)/2
        )
        plt.scatter([i + 0.2 + np.random.uniform(-0.02, 0.02) for _ in range(len(metrics['test_cv'][bacterium]))],
                    metrics['test_cv'][bacterium], color='black', s=(rcParams['lines.markersize'] ** 2)/2
                    )
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{path}_cv.png')
    plt.close()
    if mlops == 'neptune':
        run[f'{category}/{metric}_cv'].upload(f'{path}_cv.png')
    elif mlops == 'dvclive':
        run.log_artifact(f'{path}_cv.png')


def make_total_graph_total_batch(valid_metrics, test_metrics, metric, title, path, run, mlops='neptune'):
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
    plt.close()
    if mlops == 'dvclive':
        run.log_artifact(path)
    elif mlops == 'neptune':
        run[f'Batches/{metric}'].upload(path)


def make_total_graph_total_bact(valid_metrics, test_metrics, metric, title, path, run, bacteria_cols, mlops='neptune'):
    # Prepare data for plotting
    metric_df = pd.DataFrame({
        'Pathogens': bacteria_cols,
        'Valid': [valid_metrics[b] for b in bacteria_cols],
        'Test': [test_metrics[b] for b in bacteria_cols]
    })

    metric_melted_df = metric_df.melt(id_vars='Pathogens', var_name='Group', value_name=metric)
    metric_melted_df['Pathogens'] = 'Pathogens'

    # Plot metric by bacterium
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Pathogens', y=metric, hue='Group', data=metric_melted_df)
    plt.xticks(rotation=90)
    plt.title(title)
    # Remove x axis label
    plt.xlabel('')
    # Remove ticks
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    if mlops == 'dvclive':
        run.log_artifact(path)
    elif mlops == 'neptune':
        run[f'Pathogens/{metric}'].upload(path)


def plot_bars(args, run, bacteria_cols, mlops='neptune'):
    # Load the datasets
    path = f'results/multi/mz{args.mz}/rt{args.rt}/ms{args.ms_level}/' \
            f'{args.spd}spd/thr{args.threshold}/{args.mode}/{args.train_on}/{args.exp_name}/' \
            f'{args.model_name}/saved_models/'
    valid_df = pd.read_csv(f'{path}/{args.model_name}_valid_individual_results.csv', index_col=0)
    test_df = pd.read_csv(f'{path}/{args.model_name}_test_individual_results.csv', index_col=0)

    for metric in ['TPR', 'TNR', 'accuracy', 'precision', 'MCC']:
        if metric == 'TPR':
            total_valid = valid_df[valid_df['labels'] == valid_df['preds']].shape[0] / valid_df.shape[0]
            total_test = test_df[test_df['labels'] == test_df['preds']].shape[0] / test_df.shape[0]
            valid_metrics_bact, valid_metrics_bacts, valid_metrics_bacts_cv =\
                calculate_tpr_per_bact(valid_df, bacteria_cols)
            test_metrics_bact, test_metrics_bacts, test_metrics_bacts_cv =\
                calculate_tpr_per_bact(test_df, bacteria_cols)
            valid_metrics_batch, valid_metrics_batchs, valid_metrics_batch_cv =\
                calculate_tpr_per_batch(valid_df, valid_df.get('batches_labels', valid_df['batches']).unique())
            test_metrics_batch, test_metrics_batchs, test_metrics_batch_cv =\
                calculate_tpr_per_batch(test_df, test_df.get('batches_labels', test_df['batches']).unique())
        if metric == 'TNR':
            total_valid = valid_df[valid_df['labels'] == valid_df['preds']].shape[0] / valid_df.shape[0]
            total_test = test_df[test_df['labels'] == test_df['preds']].shape[0] / test_df.shape[0]
            valid_metrics_bact, valid_metrics_bacts, valid_metrics_bacts_cv =\
                calculate_tnr_per_bact(valid_df, bacteria_cols)
            test_metrics_bact, test_metrics_bacts, test_metrics_bacts_cv =\
                calculate_tnr_per_bact(test_df, bacteria_cols)
            valid_metrics_batch, valid_metrics_batchs, valid_metrics_batch_cv =\
                calculate_tnr_per_batch(valid_df, valid_df.get('batches_labels', valid_df['batches']).unique())
            test_metrics_batch, test_metrics_batchs, test_metrics_batch_cv =\
                calculate_tnr_per_batch(test_df, test_df.get('batches_labels', test_df['batches']).unique())
        elif metric == 'accuracy':
            total_valid = ACC(valid_df['labels'], valid_df['preds'])
            total_test = ACC(test_df['labels'], test_df['preds'])
            valid_metrics_bact, valid_metrics_bacts, valid_metrics_bacts_cv =\
                calculate_acc_per_bact(valid_df, bacteria_cols)
            test_metrics_bact, test_metrics_bacts, test_metrics_bacts_cv =\
                calculate_acc_per_bact(test_df, bacteria_cols)
            valid_metrics_batch, valid_metrics_batchs, valid_metrics_batch_cv =\
                calculate_acc_per_batch(valid_df, valid_df.get('batches_labels', valid_df['batches']).unique())
            test_metrics_batch, test_metrics_batchs, test_metrics_batch_cv =\
                calculate_acc_per_batch(test_df, test_df.get('batches_labels', test_df['batches']).unique())
        elif metric == 'precision':
            total_valid = valid_df[valid_df['labels'] == valid_df['preds']].shape[0] / valid_df.shape[0]
            total_test = test_df[test_df['labels'] == test_df['preds']].shape[0] / test_df.shape[0]
            valid_metrics_bact, valid_metrics_bacts, valid_metrics_bacts_cv =\
                calculate_precision_per_bact(valid_df, bacteria_cols)
            test_metrics_bact, test_metrics_bacts, test_metrics_bacts_cv =\
                calculate_precision_per_bact(test_df, bacteria_cols)
            valid_metrics_batch, valid_metrics_batchs, valid_metrics_batch_cv =\
                calculate_precision_per_batch(valid_df, valid_df.get('batches_labels', valid_df['batches']).unique())
            test_metrics_batch, test_metrics_batchs, test_metrics_batch_cv =\
                calculate_precision_per_batch(test_df, test_df.get('batches_labels', test_df['batches']).unique())
        elif metric == 'MCC':
            total_valid = MCC(valid_df['labels'], valid_df['preds'])
            total_test = MCC(test_df['labels'], test_df['preds'])
            valid_metrics_bact, valid_metrics_bacts, valid_metrics_bacts_cv =\
                calculate_mcc_per_bact(valid_df, bacteria_cols)
            test_metrics_bact, test_metrics_bacts, test_metrics_bacts_cv =\
                calculate_mcc_per_bact(test_df, bacteria_cols)
            valid_metrics_batch, valid_metrics_batchs, valid_metrics_batch_cv =\
                calculate_mcc_per_batch(valid_df, valid_df.get('batches_labels', valid_df['batches']).unique())
            test_metrics_batch, test_metrics_batchs, test_metrics_batch_cv =\
                calculate_mcc_per_batch(test_df, test_df.get('batches_labels', test_df['batches']).unique())

        metrics_bact = {
            'valid': valid_metrics_bact,
            'test': test_metrics_bact,
            'valids': valid_metrics_bacts,
            'tests': test_metrics_bacts,
            'valid_cv': valid_metrics_bacts_cv,
            'test_cv': test_metrics_bacts_cv,
        }
        metrics_batch = {
            'valid': valid_metrics_batch,
            'test': test_metrics_batch,
            'valids': valid_metrics_batchs,
            'tests': test_metrics_batchs,
            'valid_cv': valid_metrics_batch_cv,
            'test_cv': test_metrics_batch_cv,
        }
        # save valid_metrics_batch and valid_metrics_batch_cv
        # valid_metrics_batch_df = pd.DataFrame(valid_metrics_batch)
        # valid_metrics_batch_df.to_csv(f'{path}/valid_{args.exp_name}_{metric}_by_batch.csv')
        valid_metrics_batch_cv_df = pd.DataFrame(valid_metrics_batch_cv)
        valid_metrics_batch_cv_df.to_csv(f'{path}/valid_{args.exp_name}_{metric}_by_batch_cv.csv')
        # save test_metrics_batch and test_metrics_batch_cv
        # test_metrics_batch_df = pd.DataFrame(test_metrics_batch)
        # test_metrics_batch_df.to_csv(f'{path}/test_{args.exp_name}_{metric}_by_batch.csv')
        test_metrics_batch_cv_df = pd.DataFrame(test_metrics_batch_cv)
        test_metrics_batch_cv_df.to_csv(f'{path}/test_{args.exp_name}_{metric}_by_batch_cv.csv')

        try:
            make_graph(metrics_bact, metric, 'Bacterium', bacteria_cols,
                   f'{metric} by Bacterium. Total valid: {total_valid:.2f}, test: {total_test:.2f}',
                   f'{path}/{args.exp_name}_{metric}_by_bact', run, mlops
                   )
        except Exception as e:
            print(e)
            print('Error in make_graph for bacterium')
            pass
        try:
            make_graph(metrics_batch, metric, 'Batch', valid_df['batches_labels'].unique(),
                       f'{metric} by Batch. Total valid: {total_valid:.2f}, test: {total_test:.2f}',
                       f'{path}/{args.exp_name}_{metric}_by_batch', run, mlops
                       )
        except Exception as e:
            print(e)
            print('Error in make_graph for batch')
            pass
        try:
            make_total_graph_total_batch(valid_metrics_batch, test_metrics_batch, metric,
                                         f'{metric} by Batch. Total valid: {total_valid:.2f}, test: {total_test:.2f}',
                                         f'{path}/{args.exp_name}_{metric}_by_batch_total.png', run, mlops
                                         )
        except Exception as e:
            print(e)
            print('Error in make_total_graph_total_batch for batch')
        try:
            make_total_graph_total_bact(valid_metrics_bact, test_metrics_bact, metric,
                                        f'{metric} by Bacterium. Total valid: {total_valid:.2f},'
                                        f'test: {total_test:.2f}',
                                        f'{path}/{args.exp_name}_{metric}_by_bact_total.png', run, bacteria_cols, mlops
                                        )
        except Exception as e:
            print(e)
            print('Error in make_total_graph_total_bact for bacteria')
