import os
import matplotlib.pyplot as plt
import numpy as np
import itertools

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline
import torch
import copy
import torch.nn.functional as F
from scipy.stats import zscore
# split_sparse
from features_selection_sparse import split_sparse, MultiKeepNotFunctionsSparse, keep_not_zeros_sparse
import multiprocessing

def augment_data(X_train, n_aug, p=0, g=0):
    torch.manual_seed(42)
    X_train2 = copy.deepcopy(X_train)
    # y_train2 = np.array([])
    # X_train2 = np.array([])

    if n_aug > 0:
        for _ in range(n_aug):
            tmp = copy.deepcopy(X_train) + g * np.random.normal(0, 1, X_train.shape).astype(np.float32)
            tmp = F.dropout(torch.Tensor(tmp.to_numpy()), p).detach().cpu().numpy()
            if len(X_train2) > 0:
                X_train2 = np.concatenate([X_train2, tmp], 0)
            else:
                X_train2 = tmp
        columns = X_train.columns
        train_indices = np.array(X_train.index)
        # Check for duplicated indices
        if np.sum(X_train.index.duplicated()) > 0:
            duplicated_indices = np.argwhere(X_train.index.duplicated())[0]
            for ind in duplicated_indices:
                train_indices[ind] = f"{train_indices[ind]}_0"

        # train_indices = np.concatenate([train_indices] * (n_aug + 1))
        # Add copy num, 0 for original
        train_indices = [f"{x}_{i}" for i in range(n_aug + 1) for x in train_indices]

        assert len(train_indices) == np.unique(train_indices).shape[0]

        X_train2 = pd.DataFrame(X_train2, columns=columns, index=train_indices)

    return X_train2

def augment_data2(X_train, n_aug, p=0, g=0):
    torch.manual_seed(42)
    # X_train2 = copy.deepcopy(X_train)
    # y_train2 = np.array([])
    X_train2 = np.array([])

    if n_aug > 0:
        for _ in range(n_aug):
            tmp = copy.deepcopy(X_train) + g * np.random.normal(0, 1, X_train.shape).astype(np.float32)
            tmp = F.dropout(torch.Tensor(tmp.to_numpy()), p).detach().cpu().numpy()
            if len(X_train2) > 0:
                X_train2 = np.concatenate([X_train2, tmp], 0)
            else:
                X_train2 = tmp
    return X_train2


def to_categorical(y, num_classes):
    """
    1-hot encodes a tensor
    Args:
        y: values to encode
        num_classes: Number of classes. Length of the 1-encoder

    Returns:
        Tensor corresponding to the one-hot encoded classes
    """
    return torch.eye(num_classes, dtype=torch.int)[y]


def scale_data(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    scaler = None
    if scale == 'binarize':
        for group in list(data['inputs'].keys()):
            data['inputs'][group] = data['inputs'][group]
            data['inputs'][group][data['inputs'][group] > 0] = 1
            data['inputs'][group][data['inputs'][group] <= 0] = 0

    elif scale == 'robust_per_batch':
        scalers = {b: RobustScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]

    elif scale == 'standard_per_batch':
        scalers = {b: StandardScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]

    elif scale == 'minmax_per_batch':
        scalers = {b: MinMaxScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]

    elif scale == 'robust':
        scaler = RobustScaler()
        for data_type in ['inputs']:
            try:
                scaler.fit(pd.concat((data[data_type]['all'], data[data_type]['urinespositives'])))
            except:
                scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'robust_minmax':
        scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard':
        scaler = StandardScaler()
        for data_type in ['inputs']:
            try:
                scaler.fit(pd.concat((data[data_type]['all'], data[data_type]['urinespositives'])))
            except:
                scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard_minmax':
        scaler = Pipeline([('standard', StandardScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'minmax':
        scaler = MinMaxScaler()
        for data_type in ['inputs']:
            try:
                scaler.fit(pd.concat((data[data_type]['all'], data[data_type]['urinespositives'])))
            except:
                scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'minmax2':
        for data_type in ['inputs']:
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    for i in range(data[data_type][group].shape[0]):
                        # if dtype is pandas
                        try:
                            row_max = np.max(np.abs(data[data_type][group].iloc[i]))
                            if row_max != 0:
                                data[data_type][group].iloc[i] /= row_max
                        except:
                            row_max = np.max(np.abs(data[data_type][group][i]))
                            if row_max != 0:
                                data[data_type][group][i] /= row_max

    elif scale == 'zscore':
        for data_type in ['inputs']:
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    for i in range(data[data_type][group].shape[0]):
                        try:
                            data[data_type][group].iloc[i] = zscore(data[data_type][group].iloc[i])
                        except:
                            data[data_type][group][i] = zscore(data[data_type][group][i])

    elif scale == 'l1_minmax':
        scaler = Pipeline([('l1', Normalizer(norm='l1')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2_minmax':
        scaler = Pipeline([('l2', Normalizer(norm='l2')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l1':
        scaler = Pipeline([('l1', Normalizer(norm='l1'))])
        for data_type in ['inputs']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2':
        scaler = Pipeline([('l2', Normalizer(norm='l2'))])
        for data_type in ['inputs']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'none':
        return data, 'none'

    return data, scaler

def columns_stats_over0(df, name, bins, inference=False):
    """
    This column takes a pandas DataFrame and returns the number of non-zero columns
    per binning type
    First binning type: mz_parents
    Second binning type: mz_children
    Third binning type: rt
    Args:
        df: pandas DataFrame
    
    """
    # First get all columns and split on _. Keep in 3 different lists
    mz_parents = []
    mz_children = []
    rt = []
    for col in df.columns:
        features_binning = col.split('_')
        mz_parents += [features_binning[0]]
        mz_children += [features_binning[2]]
        rt += [features_binning[1]]
    mz_parents = list(set(mz_parents))
    mz_children = list(set(mz_children))
    rt = list(set(rt))
    # Make dicts to store the counts
    mz_parents_counts = {x: 0 for x in mz_parents}
    mz_children_counts = {x: 0 for x in mz_children}
    rt_counts = {x: 0 for x in rt}
    # Count the number of columns per binning type
    for col in df.columns:
        features_binning = col.split('_')
        if sum(df[col]) > 0:
            mz_parents_counts[features_binning[0]] += 1
            mz_children_counts[features_binning[2]] += 1
            rt_counts[features_binning[1]] += 1
    # Sort the dicts by keys value
    mz_parents_counts = dict(sorted(mz_parents_counts.items(), key=lambda item: float(item[0]), reverse=False))
    mz_children_counts = dict(sorted(mz_children_counts.items(), key=lambda item: int(item[0]), reverse=False))
    rt_counts = dict(sorted(rt_counts.items(), key=lambda item: int(item[0]), reverse=False))

    # Plot the counts
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].bar(mz_parents_counts.keys(), mz_parents_counts.values())
    ax[0].set_title('Mz parents')
    ax[1].bar(mz_children_counts.keys(), mz_children_counts.values())
    ax[1].set_title('Mz children')
    ax[2].bar(rt_counts.keys(), rt_counts.values())
    ax[2].set_title('RT')
    # x axius 45 degrees
    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=45)
    # For RT, only display every 10th value
    mz_keys = list(mz_children_counts.keys())
    ax[1].set_xticks([i for i in range(len(mz_keys)) if i % bins['mz'] == 0])
    ax[1].set_xticklabels([mz_keys[i] for i in range(len(mz_keys)) if i % bins['mz'] == 0], rotation=45)

    plt.tight_layout()

    os.makedirs(f"columns/{name['scaler']}/mz{name['mz']}/rt{name['rt']}/" \
                f"mz_min{name['mz_min']}/rt_min{name['rt_min']}/" \
                f"mz_max{name['mz_max']}/rt_max{name['rt_max']}/" \
                f"mz_bin{name['mz_bin']}/rt_bin{name['rt_bin']}/", exist_ok=True)
    
    plt.savefig(f"columns/{name['scaler']}/mz{name['mz']}/rt{name['rt']}/" \
                f"mz_min{name['mz_min']}/rt_min{name['rt_min']}/" \
                f"mz_max{name['mz_max']}/rt_max{name['rt_max']}/" \
                f"mz_bin{name['mz_bin']}/rt_bin{name['rt_bin']}/" \
                f"stats_over0_inference{inference}.png")


def columns_stats_0(df, name, bins, inference=False):
    """
    This column takes a pandas DataFrame and returns the number of non-zero columns
    per binning type
    First binning type: mz_parents
    Second binning type: mz_children
    Third binning type: rt
    Args:
        df: pandas DataFrame
    
    """
    # First get all columns and split on _. Keep in 3 different lists
    mz_parents = []
    mz_children = []
    rt = []
    for col in df.columns:
        features_binning = col.split('_')
        mz_parents += [features_binning[0]]
        mz_children += [features_binning[2]]
        rt += [features_binning[1]]
    mz_parents = list(set(mz_parents))
    mz_children = list(set(mz_children))
    rt = list(set(rt))
    # Make dicts to store the counts
    mz_parents_counts = {x: 0 for x in mz_parents}
    mz_children_counts = {x: 0 for x in mz_children}
    rt_counts = {x: 0 for x in rt}
    # Count the number of columns per binning type
    for col in df.columns:
        features_binning = col.split('_')
        if sum(df[col]) == 0:
            mz_parents_counts[features_binning[0]] += 1
            mz_children_counts[features_binning[2]] += 1
            rt_counts[features_binning[1]] += 1
    # Sort the dicts by keys value
    mz_parents_counts = dict(sorted(mz_parents_counts.items(), key=lambda item: float(item[0]), reverse=False))
    mz_children_counts = dict(sorted(mz_children_counts.items(), key=lambda item: int(item[0]), reverse=False))
    rt_counts = dict(sorted(rt_counts.items(), key=lambda item: int(item[0]), reverse=False))

    # Plot the counts
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].bar(mz_parents_counts.keys(), mz_parents_counts.values())
    ax[0].set_title('Mz parents')
    ax[1].bar(mz_children_counts.keys(), mz_children_counts.values())
    ax[1].set_title('Mz children')
    ax[2].bar(rt_counts.keys(), rt_counts.values())
    ax[2].set_title('RT')
    # x axius 45 degrees
    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=45)
    # For RT, only display every 10th value
    rt_keys = list(rt_counts.keys())
    ax[1].set_xticks([i for i in range(len(rt_keys)) if i % bins['mz'] == 0])
    ax[1].set_xticklabels([rt_keys[i] for i in range(len(rt_keys)) if i % bins['mz'] == 0], rotation=45)

    plt.tight_layout()    
    os.makedirs(f"columns/{name['scaler']}/mz{name['mz']}/rt{name['rt']}/" \
                f"mz_min{name['mz_min']}/rt_min{name['rt_min']}/" \
                f"mz_max{name['mz_max']}/rt_max{name['rt_max']}/" \
                f"mz_bin{name['mz_bin']}/rt_bin{name['rt_bin']}/", exist_ok=True)
    
    plt.savefig(f"columns/{name['scaler']}/mz{name['mz']}/rt{name['rt']}/" \
                f"mz_min{name['mz_min']}/rt_min{name['rt_min']}/" \
                f"mz_max{name['mz_max']}/rt_max{name['rt_max']}/" \
                f"mz_bin{name['mz_bin']}/rt_bin{name['rt_bin']}/" \
                f"stats_0_inference{inference}.png")


def scale_data_images(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    scaler = None
    if scale == 'binarize':
        for group in list(data['inputs'].keys()):
            data['inputs'][group] = data['inputs'][group]
            data['inputs'][group][data['inputs'][group] > 0] = 1
            data['inputs'][group][data['inputs'][group] <= 0] = 0

    elif scale == 'standard':
        for data_type in ['inputs']:
            data[data_type]['all'] = data[data_type]['all'] - data[data_type]['all'].mean(axis=(1, 2), keepdims=True)
            data[data_type]['all'] = data[data_type]['all'] / data[data_type]['all'].std(axis=(1, 2), keepdims=True)
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    data[data_type][group] = data[data_type][group] - data[data_type][group].mean(axis=(1, 2),
                                                                                                  keepdims=True)
                    data[data_type][group] = data[data_type][group] / data[data_type][group].std(axis=(1, 2),
                                                                                                 keepdims=True)

    elif scale == 'none':
        return data, 'none'

    return data, scaler


def scale_data_per_batch(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    if scale == 'robust':
        scalers = {b: RobustScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]

    elif scale == 'standard':
        scalers = {b: StandardScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]

    elif scale == 'minmax':
        scalers = {b: MinMaxScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]

    elif scale == 'none':
        return data

    return data, scaler


def plot_confusion_matrix(cm, class_names, acc):
    """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
    figure = plt.figure(figsize=(8, 8))
    cm_normal = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm_normal[np.isnan(cm_normal)] = 0
    plt.imshow(cm_normal, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix (acc: {acc})")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    threshold = 0.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm_normal[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return figure


def get_unique_labels(labels):
    """
    Get unique labels for a set of labels
    :param labels:
    :return:
    """
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels += [label]
    return np.array(unique_labels)


def get_combinations(cm, acc_cutoff=0.6, prop_cutoff=0.8):
    # cm2 = np.zeros(shape=(2, 2))
    to_combine = []
    for i in range(len(list(cm.index)) - 1):
        for j in range(i + 1, len(list(cm.columns))):
            acc = (cm.iloc[i, i] + cm.iloc[j, j]) / (cm.iloc[i, j] + cm.iloc[j, i] + cm.iloc[i, i] + cm.iloc[j, j])
            prop = (cm.iloc[i, j] + cm.iloc[j, i] + cm.iloc[i, i] + cm.iloc[j, j]) / (
                    np.sum(cm.iloc[i, :]) + np.sum(cm.iloc[j, :]))
            if acc < acc_cutoff and prop > prop_cutoff:
                to_combine += [(i, j)]

    # Combine all tuple that have a class in common

    new = True
    while new:
        new = False
        for i in range(len(to_combine) - 1):
            for j in range(i + 1, len(to_combine)):
                if np.sum([1 if x in to_combine[j] else 0 for x in to_combine[i]]) > 0:
                    new_combination = tuple(set(to_combine[i] + to_combine[j]))
                    to_combine = list(
                        set([ele for x, ele in enumerate(to_combine) if x not in [i, j]] + [new_combination]))
                    new = True
                    break

    return to_combine


def to_csv(lists, complete_log_path, columns):
    encoded_data = {}
    encoded_batches = {}
    encoded_cats = {}
    encoded_names = {}
    for group in list(lists.keys()):
        if len(lists[group]['encoded_values']) == 0 and group == 'all':
            continue
        if len(lists[group]['encoded_values']) > 0:
            encoded_data[group] = pd.DataFrame(np.concatenate(lists[group]['encoded_values']),
                                               # index=np.concatenate(lists[group]['labels']),
                                               ).round(4)
            encoded_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['labels']), columns=['labels']), encoded_data[group]), 1)
            encoded_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['domains']), columns=['batches']), encoded_data[group]), 1)
            encoded_data[group].index = np.concatenate(lists[group]['names'])
            encoded_batches[group] = np.concatenate(lists[group]['domains'])
            encoded_cats[group] = np.concatenate(lists[group]['classes'])
            encoded_names[group] = np.concatenate(lists[group]['names'])
        else:
            encoded_data[group] = pd.DataFrame(
                np.empty(shape=(0, encoded_data['train'].shape[1]), dtype='float')).round(4)
            encoded_batches[group] = np.array([])
            encoded_cats[group] = np.array([])
            encoded_names[group] = np.array([])

    rec_data = {}
    rec_batches = {}
    rec_cats = {}
    rec_names = {}
    for group in list(lists.keys()):
        if len(lists[group]['rec_values']) == 0 and group == 'all':
            continue
        if len(lists[group]['rec_values']) > 0:
            rec_data[group] = pd.DataFrame(np.concatenate(lists[group]['rec_values']),
                                           # index=np.concatenate(lists[group]['names']),
                                           columns=list(columns)  # + ['gender', 'age']
                                           ).round(4)

            rec_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['labels']), columns=['labels']), rec_data[group]), 1)
            rec_data[group] = pd.concat((pd.DataFrame(np.concatenate(lists[group]['domains']), columns=['batches']), rec_data[group]), 1)
            rec_data[group].index = np.concatenate(lists[group]['names'])
            rec_batches[group] = np.concatenate(lists[group]['domains'])
            rec_cats[group] = np.concatenate(lists[group]['classes'])
            rec_names[group] = np.concatenate(lists[group]['names'])
        else:
            rec_data[group] = pd.DataFrame(np.empty(shape=(0, rec_data['train'].shape[1]), dtype='float')).round(4)
            rec_batches[group] = np.array([])
            rec_cats[group] = np.array([])
            rec_names[group] = np.array([])

    rec_data = {
        "inputs": rec_data,
        "cats": rec_cats,
        "batches": rec_batches,
    }
    enc_data = {
        "inputs": encoded_data,
        "cats": encoded_cats,
        "batches": encoded_batches,
    }
    try:
        rec_data['inputs']['all'].to_csv(f'{complete_log_path}/recs.csv')
        enc_data['inputs']['all'].to_csv(f'{complete_log_path}/encs.csv')
    except:
        pd.concat((rec_data['inputs']['train'], rec_data['inputs']['valid'], rec_data['inputs']['test'])).to_csv(
            f'{complete_log_path}/recs.csv')
        pd.concat((enc_data['inputs']['train'], enc_data['inputs']['valid'], enc_data['inputs']['test'])).to_csv(
            f'{complete_log_path}/encs.csv')
    return rec_data, enc_data


def get_empty_lists():
    return {
        'preds': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'names': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'proba': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'labels': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'acc': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'mcc': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'classes': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'binary_preds': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'binary_classes': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'binary_labels': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'binary_proba': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'binary_batches': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'bact_preds': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'bact_classes': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'bact_labels': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'bact_proba': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'bact_batches': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'bact_acc': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'bact_mcc': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'inds': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'batches': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'batches_labels': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'manips': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'urines': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'concs': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        },
        'unique_batches': {
            'train': [],
            'valid': [],
            'test': [],
            'posurines': [],
        }
    }


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def remove_zero_cols(data, threshold):
    columns = data.columns
    dframe_list = split_sparse(data.values, cols_per_split=int(1e5), columns=columns)

    n_cpus = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(int(n_cpus), maxtasksperchild=1)

    try:
        assert len(dframe_list[0]) == len(dframe_list[1])
    except AssertionError:
        print(len(dframe_list[0]), len(dframe_list[1]))
        exit('Columns and dataframes are not the same length') 

    fun = MultiKeepNotFunctionsSparse(keep_not_zeros_sparse, data=dframe_list[0], 
                                      cols=dframe_list[1], 
                                      nums=dframe_list[2],
                                      threshold=threshold, 
                                      # n_processes=np.ceil(data.shape[1] / int(1e5))
                                      n_processes=1
                                      )
    notzeros = pool.map(fun.process, range(len(dframe_list[0])))

    new_columns = np.array([x for x in np.concatenate([x[0] for x in notzeros])])
    not_zeros_col = np.array([x for x in np.concatenate([x[1] for x in notzeros])])
    # not_zeros_col = np.array([x for x in np.concatenate([x[1] for x in notzeros])])
    data_matrix = data.iloc[:, not_zeros_col]

    pool.close()
    pool.join()

    return not_zeros_col


