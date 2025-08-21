import glob
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, f_classif, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from scipy.sparse import csc_matrix

def get_feature_selection_method(model_name):
    if model_name == 'mutual_info_classif':
        model = mutual_info_classif
    elif model_name == 'f_classif':
        model = f_classif
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
    return df


def keep_only_not_zeros_sparse(df, columns, nums, threshold=0):
    df = df.toarray()
    not_zeros_cols = (df != 0).any(axis=0)
    if sum(not_zeros_cols) > 0:
        cols = [i for i, c in enumerate(not_zeros_cols) if c]
        try:
            columns = [columns[c] for c in cols]
            nums = [nums[c] for c in cols]
        except:
            columns = []
            nums = []
        # df = df[:, cols]
    else:
        # df = np.empty(shape=[df.shape[0], 0])
        cols = []
        columns = []
        nums = []
    # df = csc_matrix(df)
    del df

    try:
        assert len(columns) == len(nums)
    except AssertionError as err:
        print(err, "\nOops! The list of columns and nums don't have the same length.")

    # return df, columns
    return columns, nums


def keep_not_zeros(data, threshold=0.3):
    """
    Removes columns that are all zeros
    :param data:
    :return:
    """
    # data = data.sparse.to_dense()
    not_zeros_cols = np.array(
        [i for i in range(data.shape[1]) if sum(data.iloc[:, i] > 0) / data.shape[0] >= threshold]
    )
    # print("Removing zeros...")
    data = data.iloc[:, not_zeros_cols]
    # data = data.astype(pd.SparseDtype("float64", 0))
    return data


def keep_not_zeros_sparse(data, columns, nums, threshold=0.3):
    """
    Removes columns that are all zeros
    :param data:
    :return:
    """
    data = data.toarray()
    not_zeros_cols = np.array(
        [i for i in range(data.shape[1]) if sum(data[:, i] > 0) / data.shape[0] >= threshold]
    )
    # print("Removing zeros...")
    if len(not_zeros_cols) > 0:
        # data = data[:, not_zeros_cols]
        try:
            columns = [columns[c] for c in not_zeros_cols]
            nums = [nums[c] for c in not_zeros_cols]
        except:
            print('PROBLEM:', data.shape, len(not_zeros_cols), len(columns), len(nums))
            columns = []
            nums = []
    else:
        # data = np.empty(shape=[data.shape[0], 0])
        columns = []
        nums = []
    # data = csc_matrix(data)
    del data
    return columns, nums


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


def process_sparse_data(data, cats, columns, model, dirname, args):
    if args.k == -1:
        args.k = data.shape[1]
    datasum = data.sum(0)
    inds_zeros = [i for i in range(datasum.shape[1]) if datasum[0, i] == 0]

    print(f"Out of {data.shape[1]} features, {len(inds_zeros)} columns are only zeros")

    not_zeros_cols = []
    not_zeros_cols.extend([i for i in range(datasum.shape[1]) if datasum[0, i] > 0])
    data = data[:, not_zeros_cols]
    columns = [columns[c] for c in not_zeros_cols]

    # data['pool'] = data['pool'][data['pool'].columns[not_zeros_cols]]
    # if data['test'] is not None:
    #     data['test'] = data['test'][data['test'].columns[not_zeros_cols]]

    # assert all columns with only 0s are removed
    assert len([i for i in range(datasum.shape[1]) if datasum[0, i] == 0]) == 0
    # data.to_csv('train_df.csv')
    dframe_list = split_sparse(data, columns, cols_per_split=int(1e3))

    process = Process(model, dframe_list[0], cats, dframe_list[1], np.ceil(data.shape[1] / int(1e3)))

    # TODO This was probably because an error occured with too many processes. Check if it's still necessary
    pool = multiprocessing.Pool(int(multiprocessing.cpu_count() / 10))

    mi = pd.concat(
        pool.map(process.process,
                 range(len(dframe_list[0]))
                 )
    )

    top_indices = np.argsort(mi.values.reshape(-1))[::-1]
    top_scores = np.sort(mi.values.reshape(-1))[::-1]
    top_k_indices = top_indices[:args.k].reshape(-1)
    top_k_scores = top_scores[:args.k].reshape(-1)
    top_k_columns = np.array(columns)[top_k_indices]

    features_scores = pd.DataFrame(top_k_scores.reshape([-1, 1]),
                                   columns=['score'],
                                   index=top_k_columns)
    os.makedirs(f'{dirname}/{args.run_name}/', exist_ok=True)
    features_scores.to_csv(
        f'{dirname}/{args.run_name}/{args.feature_selection}_scores.csv',
        index_label='minp_maxp_rt_mz'
    )


def process_sparse_data_supervised(data, cats, batches, columns, model, dirname, args, inference=False):
    # TODO POOLS HANDLING
    if args.k == -1:
        args.k = data.shape[1]
    datasum = data.sum(0)
    inds_zeros = [i for i in range(datasum.shape[1]) if datasum[0, i] == 0]

    print(f"Out of {data.shape[1]} features, {len(inds_zeros)} columns are only zeros")

    not_zeros_cols = []
    not_zeros_cols.extend([i for i in range(datasum.shape[1]) if datasum[0, i] > 0])
    data = data[:, not_zeros_cols]
    columns = [columns[c] for c in not_zeros_cols]

    # data['pool'] = data['pool'][data['pool'].columns[not_zeros_cols]]
    # if data['test'] is not None:
    #     data['test'] = data['test'][data['test'].columns[not_zeros_cols]]

    # assert all columns with only 0s are removed
    if not inference:
        assert len([i for i in range(datasum.shape[1]) if datasum[0, i] == 0]) == 0
    # data.to_csv('train_df.csv')
    features_scores = [None for _ in range(args.n_splits)]

    for i in range(args.n_splits):
        if args.groupkfold:
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=i)
            train_nums = np.arange(0, data.shape[0])
            splitter = skf.split(train_nums, cats, batches)
            # train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
            train_nums = np.arange(0, data.shape[0])
            splitter = skf.split(train_nums, cats)
        # train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
        _, valid_inds = splitter.__next__()
        _, test_inds = splitter.__next__()
        train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]
        
        dframe_list = split_sparse(data[train_inds], columns, cols_per_split=int(1e3))

        process = Process(model, dframe_list[0], cats[train_inds], dframe_list[1], np.ceil(data.shape[1] / int(1e3)))

        # TODO If there is an error, there might be a problem with the number of processes
        pool = multiprocessing.Pool(int(multiprocessing.cpu_count() - 2))

        mi = pd.concat(
            pool.map(process.process,
                    range(len(dframe_list[0]))
                    )
        )

        top_indices = np.argsort(mi.values.reshape(-1))[::-1]
        top_scores = np.sort(mi.values.reshape(-1))[::-1]
        top_k_indices = top_indices[:args.k].reshape(-1)
        top_k_scores = top_scores[:args.k].reshape(-1)
        # top_columns = data.columns[top_indices]
        top_k_columns = np.array(columns)[top_k_indices]

        features_scores[i] = pd.DataFrame(top_k_scores.reshape([-1, 1]),
                                    columns=['score'],
                                    index=top_k_columns)
        os.makedirs(f'{dirname}/{args.run_name}/', exist_ok=True)
        features_scores[i].to_csv(
            f'{dirname}/{args.run_name}/{args.feature_selection}_scores_{i}.csv',
            index_label='minp_maxp_rt_mz'
        )
        inds = {
            'train': train_inds,
            'valid': valid_inds,
            'test': test_inds
        }
        pickle.dump(inds, open(f'{dirname}/{args.run_name}/indices_{i}.pkl', 'wb'))


    # concat all features scores after changing the order based on features_scores[0]
    features_scores = pd.concat(
        [features_scores[i].reindex(features_scores[0].index) for i in range(args.n_splits)],
        axis=1
    )
    features_scores = features_scores.mean(axis=1)
    features_scores = features_scores.sort_values(ascending=False)
    features_scores.to_csv(
        f'{dirname}/{args.run_name}/{args.feature_selection}_scores.csv',
        index_label='minp_maxp_rt_mz'
    )
    
    
    # TODO save a version of the features scores averaged over all splits and ordered by best scores
    

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
    tsvs_list = []
    for _, file in enumerate(tsv_list):
        if 'hela' in file or 'Hela' in file:
            continue
        if len(file.split('\\')) > 1:
            sample = file.split('\\')[-1].split('.')[0]
        else:
            sample = file.split('/')[-1].split('.')[0]

        samples.append(sample)
        tmp = sample.split('_')
        batch = tmp[0]
        manip = tmp[3]
        label = tmp[2]
        if label in ['blanc', 'Blanc', 'blk', 'Blk', 'urinespositives'] or label == 'Blanc':
            concentration = 'NA'
            urine = tmp[5]
        elif tmp[-1] in ['cultpure']:
            concentration = 'NA'
            urine = 'NA'
        else:
            concentration = tmp[7]
            urine = tmp[6]

        # label = f"{batch}_{'_'.join(tmp[-3:])}".lower()
        label = f"{label}_{batch}_{manip}_{urine}_{concentration}".lower()
        labels_list.append(label)
        tsvs_list.append(file)
    del tsv_list
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
        "tsv": tsvs_list,
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


def split_sparse(dframe, columns, cols_per_split):
    n_partitions = int(np.ceil(dframe.shape[1] / cols_per_split))
    dframes_list = [
        dframe[:, x * cols_per_split:(x + 1) * cols_per_split]
        if x < n_partitions - 1 else dframe[:, x * cols_per_split:]
        for x in range(n_partitions)
    ]
    cols_list = [
        columns[x * cols_per_split:(x + 1) * cols_per_split]
        if x < n_partitions - 1 else columns[x * cols_per_split:]
        for x in range(n_partitions)
    ]
    col_nums_list = [np.arange(x * cols_per_split, (x + 1) * cols_per_split)
        if x < n_partitions - 1 else np.arange(x * cols_per_split, dframe.shape[1])
        for x in range(n_partitions)
    ]
    try:
        assert np.sum(
            [df1.shape[1] for df1 in dframes_list]) == dframe.shape[1]
    except AssertionError as err:
        print(err, "\nOops! The list of dataframes don't have the same shape as inial dataframe.")
    return dframes_list, cols_list, col_nums_list


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


class MultiKeepNotFunctionsSparse:
    """
    Class for multiprocessing of keep_not_zeros and keep_not_duplicated
    """

    def __init__(self, function, data, cols, nums, threshold, n_processes):
        """
        :param function:
        :param data:
        :return:
        """

        self.function = function
        self.data = data
        self.cols = cols
        self.nums = nums
        self.threshold = threshold
        self.n_processes = n_processes

    def process(self, i):
        """
        Process function
        """
        print(f"Process: {i}/{self.n_processes}")
        results = self.function(self.data[i], self.cols[i], self.nums[i], threshold=self.threshold)
        self.data[i] = []
        return results


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

    def __init__(self, model, data, labels, columns, n_processes):
        """
        :param model:
        :param data:
        :param labels:
        :return:
        """

        self.model = model
        self.data = data
        self.labels = labels
        self.columns = columns
        self.n_processes = n_processes
        self.unique_labels = np.unique(labels)
        self.cats = np.array([np.argwhere(self.unique_labels == label)[0][0] for label in labels])

    def process(self, i):
        """
        Process function
        """
        print(f"Process: {i}/{self.n_processes}")
        # results = self.model(self.data[i])
        data = self.data[i].toarray()
        if self.model == VarianceThreshold:
            model = self.model(threshold=0)
            _ = model.fit_transform(data)
            results = model.variances_
        else:
            results = self.model(data, self.labels)
            if self.model != mutual_info_classif:
                results = results[0]
        return pd.DataFrame(
            data=results,
            index=self.columns[i],
            columns=['score']
        )

    def get_n_cols(self):
        """
        Gets n columns. Mainly just to have a second class so pylint does not complain.
        """
        return len(self.data.index)