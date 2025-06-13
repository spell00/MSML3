import glob
import multiprocessing
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif, VarianceThreshold


def get_feature_selection_method(model_name):
    if model_name == 'mutual_info_classif':
        model = mutual_info_classif
    elif model_name == 'f_classif':
        model = f_classif
    elif model_name == 'variance':
        model = VarianceThreshold
    # TODO these won't work, need to implement with an estimator
    # elif model_name == 'RFE':
    #     model = RFE
    # elif model_name == 'RFECV':
    #     model = RFECV
    else:
        exit('Feature selection method unavailable. Must be one of [mutual_info_classif, f_classif, RFE, RFECV]')
    return model


def keep_only_not_zeros(df, threshold=0):
    df = df.loc[:, (df != 0).any(axis=0)]
    return df, df.columns


def keep_not_zeros(data, threshold=0.3):
    """
    Removes columns that are all zeros
    :param data:
    :return:
    """
    not_zeros_cols = np.array(
        [i for i in range(data.values.shape[1]) if sum(data.values[:, i] > 0) / len(data) > threshold]
    )
    # print("Removing zeros...")
    data = data.iloc[:, not_zeros_cols]

    return data, data.columns


def keep_not_duplicated(data, threshold=0.95):
    """
    Removes columns that are all zeros
    :param data:
    :return:
    """
    print("Finding columns with mostly duplicate values...")
    not_duplicates_cols = []
    for i in range(data.values.shape[1]):
        for val in list(set(data.values[:, i])):
            if (sum(data.values[:, i] != val) / len(data)) > threshold:
                not_duplicates_cols += [i]
    not_duplicates_cols = np.array(not_duplicates_cols)
    # if threshold > 0:
    data = data.iloc[:, not_duplicates_cols]
    # else:
    # The code above returns the same thing as above if threshold is 0, but it's faster
    #     data = data.loc[:, (data != 0).any(axis=0)]

    return data, not_duplicates_cols


def process_data(data, cats, model, dirname, feature_selection, k=10000, run_name=''):
    if k == -1:
        k = data.shape[1]

    inds_zeros = [i for i, x in enumerate(data.sum(0)) if x == 0]

    print(f"Out of {data.shape[1]} features, {len(inds_zeros)} columns are only zeros")

    not_zeros_cols = []
    not_zeros_cols.extend([i for i, sum0 in enumerate(data.sum(0)) if sum0 > 0])
    data = data[data.columns[not_zeros_cols]]
    # data['pool'] = data['pool'][data['pool'].columns[not_zeros_cols]]
    # if data['test'] is not None:
    #     data['test'] = data['test'][data['test'].columns[not_zeros_cols]]

    # assert all columns with only 0s are removed
    assert len([i for i, x in enumerate(data.sum(0)) if x == 0]) == 0
    # data.to_csv('train_df.csv')
    dframe_list = split_df(data, cols_per_split=int(1e3))

    process = Process(model, dframe_list, cats, np.ceil(data.shape[1] / int(1e3)))
    pool = multiprocessing.Pool(int(multiprocessing.cpu_count() / 10))

    mi = pd.concat(
        pool.map(process.process,
                 range(len(dframe_list))
                 )
    )

    top_indices = np.argsort(mi.values.reshape(-1))[::-1]
    top_scores = np.sort(mi.values.reshape(-1))[::-1]
    top_k_indices = top_indices[:k].reshape(-1)
    top_k_scores = top_scores[:k].reshape(-1)
    # top_columns = data.columns[top_indices]
    top_k_columns = data.columns[top_k_indices]

    features_scores = pd.DataFrame(top_k_scores.reshape([-1, 1]),
                                   columns=['score'],
                                   index=top_k_columns)
    os.makedirs(f'{dirname}/{run_name}/', exist_ok=True)
    features_scores.to_csv(
        f'{dirname}/{run_name}/{feature_selection}_scores.csv',
        index_label='minp_maxp_rt_mz')

    # data = pd.DataFrame(data.iloc[:, top_k_indices],
    #                              columns=top_k_columns,
    #                              index=data.index)
    # data['pool'] = pd.DataFrame(data['pool'].iloc[:, top_k_indices],
    #                             columns=top_k_columns,
    #                             index=data['pool'].index)

    # data.iloc[:] = np.round(np.nan_to_num(data), 2)
    # data['pool'].iloc[:] = np.round(np.nan_to_num(data['pool']), 2)

    # indices = np.argwhere(features_scores.to_numpy()[:, 0] > cutoff)[:, 0]
    # features_scores_gt = pd.DataFrame(top_k_scores[indices].reshape(-1, 1),
    #                                   columns=['score'],
    #                                   index=features_scores.index[indices])
    # features_scores_gt.to_csv(
    #     f'{dirname}/{run_name}/{feature_selection}_scores_gt{cutoff}.csv',
    #     index_label='minp_maxp_rt_mz')

    # data = data[top_k_columns[indices]]
    # data['pool'] = data['pool'][top_k_columns[indices]]

    # if combat_corr:
    #     data.to_csv(
    #         f'{dirname}/{run_name}/train_inputs_combat.csv',
    #         index=True, index_label='ID')
    #     data['pool'].to_csv(
    #         f'{dirname}/{run_name}/train_pool_inputs_combat.csv',
    #         index=True, index_label='ID')
    # else:
    #     data.to_csv(
    #         f'{dirname}/{run_name}/train_inputs.csv',
    #         index=True, index_label='ID')
    #     data['pool'].to_csv(
    #         f'{dirname}/{run_name}/train_pool_inputs.csv',
    #         index=True, index_label='ID')


def process_data_unsupervised(data, cats, model, cutoff, dirname, feature_selection, k=10000, run_name='', combat_corr=0):
    if k == -1:
        k = data["train"].shape[1]

    inds_zeros = [i for i, x in enumerate(data['train'].sum(0)) if x == 0]

    print(
        f"Out of {data['train'].shape[1]} features, {len(inds_zeros)} columns are only zeros")

    not_zeros_cols = []
    not_zeros_cols.extend([i for i, sum0 in enumerate(data['train'].sum(0)) if sum0 > 0])
    data['train'] = data['train'][data['train'].columns[not_zeros_cols]]
    data['pool'] = data['pool'][data['pool'].columns[not_zeros_cols]]
    # if data['test'] is not None:
    #     data['test'] = data['test'][data['test'].columns[not_zeros_cols]]

    # assert all columns with only 0s are removed
    assert len([i for i, x in enumerate(data['train'].sum(0)) if x == 0]) == 0
    # data['train'].to_csv('train_df.csv')
    dframe_list = split_df(data['train'], cols_per_split=int(1e3))

    process = Process(model, dframe_list, cats['train'], np.ceil(data["train"].shape[1] / int(1e3)))
    pool = multiprocessing.Pool(int(multiprocessing.cpu_count() / 10))

    mi = pd.concat(
        pool.map(process.process,
                 range(len(dframe_list))
                 )
    )

    top_indices = np.argsort(mi.values.reshape(-1))[::-1]
    top_scores = np.sort(mi.values.reshape(-1))[::-1]
    top_k_indices = top_indices[:k].reshape(-1)
    top_k_scores = top_scores[:k].reshape(-1)
    top_columns = data['train'].columns[top_indices]
    top_k_columns = data['train'].columns[top_k_indices]

    features_scores = pd.DataFrame(top_k_scores.reshape([-1, 1]),
                                   columns=['score'],
                                   index=top_k_columns)
    os.makedirs(f'{dirname}/{run_name}/', exist_ok=True)
    features_scores.to_csv(
        f'{dirname}/{run_name}/{feature_selection}_scores.csv',
        index_label='minp_maxp_rt_mz')

    data['train'] = pd.DataFrame(data['train'].iloc[:, top_k_indices],
                                 columns=top_k_columns,
                                 index=data['train'].index)
    data['pool'] = pd.DataFrame(data['pool'].iloc[:, top_k_indices],
                                columns=top_k_columns,
                                index=data['pool'].index)

    data['train'].iloc[:] = np.round(np.nan_to_num(data['train']), 2)
    data['pool'].iloc[:] = np.round(np.nan_to_num(data['pool']), 2)

    indices = np.argwhere(features_scores.to_numpy()[:, 0] > cutoff)[:, 0]
    features_scores_gt = pd.DataFrame(top_k_scores[indices].reshape(-1, 1),
                                      columns=['score'],
                                      index=features_scores.index[indices])
    features_scores_gt.to_csv(
        f'{dirname}/{run_name}/{feature_selection}_scores_gt{cutoff}.csv',
        index_label='minp_maxp_rt_mz')

    data['train'] = data['train'][top_k_columns[indices]]
    data['pool'] = data['pool'][top_k_columns[indices]]

    if combat_corr:
        data['train'].to_csv(
            f'{dirname}/{run_name}/train_inputs_combat.csv',
            index=True, index_label='ID')
        data['pool'].to_csv(
            f'{dirname}/{run_name}/train_pool_inputs_combat.csv',
            index=True, index_label='ID')
    else:
        data['train'].to_csv(
            f'{dirname}/{run_name}/train_inputs.csv',
            index=True, index_label='ID')
        data['pool'].to_csv(
            f'{dirname}/{run_name}/train_pool_inputs.csv',
            index=True, index_label='ID')


def count_array(arr):
    """
    Counts elements in array

    :param arr:
    :return:
    """
    elements_count = {}
    for element in arr:
        if element in elements_count:
            elements_count[element] += 1
        else:
            elements_count[element] = 1
    to_remove = []
    for key, value in elements_count.items():
        print(f"{key}: {value}")
        if value <= 2:
            to_remove += [key]

    return to_remove


def make_lists(dirinput, path, run_name):
    """
    Makes lists

    :param dirinput:
    :param path:
    :param run_name:
    :return:
    """
    tsv_list = glob.glob(dirinput + '*.tsv')
    # Initiate variables
    samples = []
    # pool_files = []

    # for _, file in enumerate(tsv_list):
    #     if 'pool' in file:
    #         pool_files += [file]

    # for psample in pool_files:
    #     tsv_list.remove(psample)
    labels_list = []
    for _, file in enumerate(tsv_list):
        if len(file.split('\\')) > 1:
            sample = file.split('\\')[-1].split('.')[0]
        else:
            sample = file.split('/')[-1].split('.')[0]

        samples.append(sample)
        tmp = sample.split('_')
        batch = tmp[0]
        manip = tmp[3]
        label = tmp[2]
        if label == 'blanc' or label == 'Blanc':
            concentration = 'NA'
        else:
            concentration = tmp[6]
        urine = tmp[4]

        # label = f"{batch}_{'_'.join(tmp[-3:])}".lower()
        label = f"{label}_{batch}_{manip}_{urine}_{concentration}".lower()
        labels_list.append(label)

    categories = [x.split('_')[0] for x in labels_list]

    names_df = pd.DataFrame(
        np.concatenate(
            (np.array(labels_list).reshape((-1, 1)),
             np.array(samples).reshape((-1, 1)),
             np.array(categories).reshape((-1, 1)),
             ), 1)
    )
    os.makedirs(path, exist_ok=True)
    names_df.to_csv(f'{path}/fnames_ids_{run_name}.csv', index=False,
                    header=['ID', 'fname', 'category'])
    return {
        "samples": samples,
        "tsv": tsv_list,
        "labels": labels_list,
    }


def split_df(dframe, cols_per_split):
    n_partitions = int(np.ceil(dframe.shape[1] / cols_per_split))
    dframes_list = [
        dframe.iloc[:, x * cols_per_split:(x + 1) * cols_per_split]
        if x < n_partitions - 1 else dframe.iloc[:, x * cols_per_split:]
        for x in range(n_partitions)
    ]
    try:
        assert np.sum(
            [df1.shape[1] for df1 in dframes_list]) == dframe.shape[1]
    except AssertionError as err:
        print(err, "\nOops! The list of dataframes don't have the same shape as inial dataframe.")
    return dframes_list


def make_matrix(finals, labels):
    # indices_columns = pd.MultiIndex.from_product([finals[0].index, finals[0].columns])
    new_columns = [f'{x}_{y}' for x in finals[0].index for y in finals[0].columns]
    finals = pd.concat(
        [pd.DataFrame(final.values.flatten().reshape(1, -1), columns=new_columns)
         for final in finals], 0
    )
    finals.index = labels
    return finals


def get_plates(path, labels, blk_plate=1):
    infos = pd.read_csv(path, index_col=0)
    infos.index = [x.lower() for x in infos.index]
    plates = []
    # [np.where(infos.index == label.split('_')[1])[0][0] for label in labels]
    for label in labels:
        if 'pool' in label:
            plates += [1]
        elif 'blk' not in label:
            plates += [infos['Plate'][np.where(infos.index == label.split('_')[1])[0][0]]]
        else:
            if 'blk_p' not in label:
                plates += [blk_plate]
            else:
                plate = int(label.split('_')[2].split('p')[1].split('-')[0])
                plates += [plate]
    return plates


class MultiKeepNotFunctions:
    """
    Class for multiprocessing of keep_not_zeros and keep_not_duplicated
    """

    def __init__(self, function, data, threshold, n_processes):
        """
        :param function:
        :param data:
        :return:
        """

        self.function = function
        self.data = data
        self.threshold = threshold
        self.n_processes = n_processes

    def process(self, i):
        """
        Process function
        """
        print(f"Process: {i}/{self.n_processes}")
        return self.function(self.data[i], threshold=self.threshold)


class Process:
    """
    Class for multiprocessing of feature selection
    """

    def __init__(self, model, data, labels, n_processes):
        """
        :param model:
        :param data:
        :param labels:
        :return:
        """

        self.model = model
        self.data = data
        self.labels = labels
        self.n_processes = n_processes

    def process(self, i):
        """
        Process function
        """
        print(f"Process: {i}/{self.n_processes}")
        if self.model == VarianceThreshold:
            model = self.model(threshold=0)
            _ = model.fit_transform(self.data[i])
            results = model.variances_
        else:
            results = self.model(self.data[i], self.labels)
            if self.model != mutual_info_classif:
                results = results[0]
        return pd.DataFrame(
            data=results,
            index=self.data[i].columns,
            columns=['score']
        )

    def get_n_cols(self):
        """
        Gets n columns. Mainly just to have a second class so pylint does not complain.
        """
        return len(self.data.index)
    
