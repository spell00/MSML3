#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Simon Pelletier

"""

from datetime import datetime

# import re
import os
import time
# import scipy
import logging
import multiprocessing
from multiprocessing import set_start_method
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from PIL import Image
from matplotlib import cm
from pickle import dump, load
from scipy.sparse import csr_matrix, vstack, hstack, csc_matrix
from tqdm import tqdm
# import nibabel as nib
import warnings
# import queue
from msml.preprocess.to_delete.features_selection import get_feature_selection_method
from msml.preprocess.to_delete.features_selection_sparse import keep_only_not_zeros_sparse, keep_not_zeros_sparse, \
    process_sparse_data, count_array, make_lists, split_sparse, MultiKeepNotFunctionsSparse, \
    process_sparse_data_supervised
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
from msalign import msalign

warnings.filterwarnings("ignore")
logging.basicConfig(filename='make_tensors_ms1.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

start_time = datetime.now()


class MakeTensorsMultiprocess:
    """
    Concat
    """

    def __init__(self, tsv_list, labels_list, bins, path, args):
        """
        :param tsv_list:
        :param labels_list:
        :param test_run:
        :return:
        """
        os.makedirs(f'{path}/images', exist_ok=True)
        os.makedirs(f'{path}/csv', exist_ok=True)

        self.bins = bins
        self.is_sparse = args.is_sparse

        self.path = path
        self.tsv_list = tsv_list
        self.labels_list = labels_list
        self.save = args.save
        self.test_run = args.test_run
        # self.mz_shift = mz_shift
        # self.rt_shift = rt_shift
        self.log2 = args.log2
        self.find_peaks = args.find_peaks
        self.lowess = args.lowess
        if args.n_samples != -1:
            self.tsv_list = self.tsv_list[:args.n_samples]
            self.labels_list = self.labels_list[:args.n_samples]

    def process(self, index):
        """

        This process makes a 'tensor' (list of dataframes)
        :param i:
        :return:
        """
        startTime = time.time()
        # print(len(gc.get_objects()))
        try:
            file, label = self.tsv_list[index], self.labels_list[index]
        except:
            exit('Error with tsv index')
        try:
            tsv = pd.read_csv(file, header=0, sep='\t')
            tsv = tsv.astype({c: np.float32 for c in tsv.select_dtypes(include='float64').columns})
        except:
            exit('Error reading csv')
        print(
            f"Processing file {index}: {file}")

        # if label != 'aba_01-03-2024_240222_u002_l':
        #     return None, None, None
        tsv = tsv[tsv.bin_intensity != 0]

        if self.bins['mz_shift']:
            mz_shift = self.bins['mz_bin_post'] / 2
        else:
            mz_shift = 0
        if self.bins['rt_shift']:
            rt_shift = self.bins['rt_bin_post'] / 2
        else:
            rt_shift = 0

        if self.is_sparse:
            if self.log2 == 'inloop':
                dtype = pd.SparseDtype("float32", 0)
            else:
                dtype = pd.SparseDtype("float64", 0)
        else:
            if self.log2 == 'inloop':
                dtype = "float32"
            else:
                dtype = "float64"

        final = pd.DataFrame(
            np.zeros([int(np.ceil(tsv.mz_bin.max() / self.bins['mz_bin_post'])) + 1,
                      int(np.ceil(tsv.rt_bin.max() / self.bins['rt_bin_post'])) + 1]),
            dtype=np.float32,
            columns=np.arange(0, tsv.rt_bin.max() + self.bins['rt_bin_post'], self.bins['rt_bin_post']).round(
                self.bins['rt_rounding']) - rt_shift,
            index=np.arange(0, tsv.mz_bin.max() + self.bins['mz_bin_post'], self.bins['mz_bin_post']).round(
                self.bins['mz_rounding']) - mz_shift
        )
        final.index = np.round(final.index, self.bins['mz_rounding'])
        final.columns = np.round(final.columns, self.bins['rt_rounding'])
        rt = float(final[list(final.keys())[0]][0])
        # prev_mz = -1
        for i, line in enumerate(tsv.values):
            rt, mz, intensity = line
            if np.isnan(rt) or np.isnan(mz):
                continue
            if self.bins['rt_shift']:
                tmp_rt = np.floor(np.round(rt / self.bins['rt_bin_post'], 8)) * self.bins['rt_bin_post']
                if rt % (self.bins['rt_bin_post'] / 2) > (self.bins['rt_bin_post'] / 2):
                    tmp_rt += self.bins['rt_bin_post'] / 2
                else:
                    tmp_rt -= self.bins['rt_bin_post'] / 2
                rt = tmp_rt
            elif self.bins['rt_bin_post'] != self.bins['rt_bin']:
                rt = np.round(np.round(rt / self.bins['rt_bin_post'], 8), self.bins['rt_bin_post']) * self.bins['rt_bin_post']

            if self.bins['mz_shift']:
                tmp_mz = np.floor(np.round(mz / self.bins['mz_bin_post'], 8)) * self.bins['mz_bin_post']
                if mz % (self.bins['mz_bin_post'] / 2) > (self.bins['mz_bin_post'] / 2):
                    tmp_mz += self.bins['mz_bin_post'] / 2
                else:
                    tmp_mz -= self.bins['mz_bin_post'] / 2
                mz = tmp_mz
            elif self.bins['mz_bin_post'] != self.bins['mz_bin']:
                mz = np.round(np.round(mz / self.bins['mz_bin_post'], 8), self.bins['mz_bin_post']) * self.bins['mz_bin_post']
            if self.bins['rt_rounding'] != 0:
                rt = np.round(rt, self.bins['rt_rounding'])
            if self.bins['mz_rounding'] != 0:
                mz = np.round(mz, self.bins['mz_rounding'])
            mz = np.round(float(mz), self.bins['mz_rounding'])
            rt = np.round(float(rt), self.bins['rt_rounding'])
            if self.log2 == 'inloop':
                final.loc[mz].loc[rt] += np.log1p(intensity)
            else:
                final.loc[mz].loc[rt] += intensity
            if self.test_run and i > 10000:
                break
        final = final.astype(dtype)
        if self.lowess:
            for ii, line1 in enumerate(final.values):
                final.iloc[ii] = lowess(line1, list(final.columns), frac=0.1, return_sorted=False)
        if self.find_peaks:
            for ii, line1 in enumerate(final.values):
                mask = find_peaks(final.iloc[ii], height=0.1, distance=2)
                # Make all values 0 except the ones that are in mask
                final.iloc[ii] = pd.Series([final.iloc[ii].values[i] if i in mask[0] else 0 for i in range(len(final.iloc[ii]))])
        # os.makedirs(f"{self.path}/nibabel/", exist_ok=True)
        if self.save:
            df = final
            df = df / df.max()
            _ = self.save_images_and_csv(final, df, label)
        total_memory = final.memory_usage().sum() / 2 ** 20
        total_time = (time.time() - startTime) / 60
        print(
            f"Finished file {index}. Total memory: {np.round(total_memory, 2)}  MB, time: {np.round(total_time, 2)} minutes")
        # print(len(gc.get_objects()))
        return final, list(final.keys()), label

    def n_samples(self):
        """
        Gets n samples. Mainly just to have a second class so pylint does not complain.
        """
        return len(self.tsv_list)

    def save_images_and_csv(self, final, df, label):
        os.makedirs(f"{self.path}/csv/", exist_ok=True)
        os.makedirs(f"{self.path}/images/", exist_ok=True)
        final.to_csv(f"{self.path}/csv/{label}.csv", index_label='ID')
        im = Image.fromarray(np.uint8(cm.gist_earth(df) * 255))
        im.save(f"{self.path}/images/{label}.png")
        im.close()
        del im


def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


def adjust_tensors(list_matrices, max_features, args_dict):
    # TODO VERIFY THAT THE ADJUSTMENTS ARE CORRECT; everything is appended to the high end of the tensor, is it correct?
    # MIGHT CAUSE IMPORTANT BATCH EFFECTS IF NOT RIGHT
    max_rt = max_features['max_rt']
    max_mz = max_features['max_mz']
    # TODO could be parallelized if worth it
    for j, matrices in enumerate(list_matrices):
        with tqdm(total=len(matrices), position=0, leave=True) as pbar:
            for i, matrix in enumerate(matrices):
                # matrix = data_matrix[0][0]
                # https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
                # hstack of csc matrices should be faster than coo (worst) or csr
                if max_mz - matrix.shape[0] > 0:
                    matrix = csc_matrix(
                        vstack((
                            matrix, np.zeros((int((max_mz - matrix.shape[0])), matrix.shape[1])))
                        ))
                else:
                    matrix = csc_matrix(matrix)
                if max_rt - matrix.shape[1] > 0:
                    matrix = csc_matrix(
                        hstack((
                            matrix, csc_matrix(np.zeros((matrix.shape[0], int((max_rt - matrix.shape[1])))))
                        ))
                    )
                else:
                    matrix = csc_matrix(matrix)
                matrices[i] = matrix.reshape([1, -1]).tocsr()
                del matrix
                pbar.update(1)
        list_matrices[j] = matrices
    print('Tensors are adjusted.')
    return list_matrices


def make_df(dirinput, dirname, bins, args_dict, names_to_keep=None):
    """
    Loads the tsvs an

    :param dirinput:
    :param dirname:
    :param args_dict:
    :return:
    """
    # Get list of all wanted tsv

    # path = f"{dirname}/"
    lists = make_lists(dirinput, dirname, args_dict.run_name)
    if names_to_keep is not None:
        inds_to_keep = [i for i, x in enumerate(lists['labels']) if
                        x.split('_')[1].lower() in names_to_keep]
        lists["tsv"] = np.array(lists['tsv'])[inds_to_keep].tolist()
        lists["labels"] = np.array(lists['labels'])[inds_to_keep].tolist()
    concat = MakeTensorsMultiprocess(lists["tsv"], lists["labels"], bins, dirname, args_dict)

    if args_dict.n_cpus < 1:
        n_cpus = multiprocessing.cpu_count() + args_dict.n_cpus
    else:
        n_cpus = args_dict.n_cpus

    with multiprocessing.Pool(n_cpus, maxtasksperchild=10) as pool:
        data_matrix = pool.map(concat.process,
                               range(len(concat.tsv_list))
                               )
        # , chunksize=1)
        print('Tensors are done.')
        # data_matrix.wait()
        pool.close()
        pool.terminate()
        pool.join()

    print('Tensors are done!')

    labels = [x[2].lower() for x in data_matrix]

    max_rt = max([y[0].shape[1] for y in data_matrix])
    max_mz = max([y[0].shape[0] for y in data_matrix])

    # assert type(max_rt) == int and type(max_mz) == int

    # data_matrix = np.concatenate(([x[0] for x in data_matrix]))
    if args_dict.rt_rounding == 0:
        rt_bin = int(args_dict.rt_bin_post)
    else:
        rt_bin = np.round(float(args_dict.rt_bin_post), args_dict.rt_rounding)
    if args_dict.mz_rounding == 0:
        mz_bin = int(args_dict.mz_bin_post)
    else:
        mz_bin = np.round(float(args_dict.mz_bin_post), args_dict.mz_rounding)
    # Make all matrices of equal shapes so they can be concatenated
    matrices = []
    new_labels = []

    # TODO could be parallelized if worth it
    # To save on memory, pop old matrices as new ones are built at the right size for stacking
    with tqdm(total=len(data_matrix), position=0, leave=True) as pbar:
        while len(data_matrix) > 0:
            matrix, label = data_matrix[0][0], labels[0]
            # https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
            # hstack of csc matrices should be faster than coo (worst) or csr
            if max_mz - matrix.shape[0] > 0:
                matrix = csc_matrix(
                    vstack((
                        matrix, np.zeros((int((max_mz - matrix.shape[0])), matrix.shape[1])))
                    ))
            else:
                matrix = csc_matrix(matrix)
            if max_rt - matrix.shape[1] > 0:
                matrix = csc_matrix(
                    hstack((
                        matrix, csc_matrix(np.zeros((matrix.shape[0], int((max_rt - matrix.shape[1])))))
                    ))
                )
            else:
                matrix = csc_matrix(matrix)

            matrices += [matrix]
            new_labels += [label]
            data_matrix = data_matrix[1:]
            labels = labels[1:]
            # del matrix
            pbar.update(1)

    print('Tensors are adjusted.')
    if bins['rt_shift']:
        rt_shift = rt_bin / 2
    else:
        rt_shift = 0
    if bins['mz_shift']:
        mz_shift = mz_bin / 2
    else:
        mz_shift = 0

    mzs = [np.round(mz * args_dict.mz_bin_post, args_dict.mz_rounding) - mz_shift for mz in range(max_mz)]
    rts = [np.round(rt * args_dict.rt_bin_post, args_dict.rt_rounding) - rt_shift for rt in range(max_rt)]

    return matrices, new_labels, [f"{rt}_{mz}" for rt in rts for mz
                                   in mzs], {'max_rt': max_rt, 'max_mz': max_mz}
    


if __name__ == "__main__":

    # record time
    set_start_method("spawn")
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--test_run", type=int, default=0,
                        help="Is it a test run? 1 for true 0 for false")
    parser.add_argument("--n_samples", type=int, default=-1,
                        help="How many samples to run? Only modify to make test runs faster (-1 for all samples)")
    parser.add_argument("--log2", type=str, default='after',
                        help='log the data in the loop, after the loop or not log the data. Choices: [inloop, after, no]')
    parser.add_argument("--shift", type=int, default=0, help='Shift the data matrix')
    parser.add_argument("--binary", type=int, default=0, help='Blanks vs bacteria')
    parser.add_argument("--threshold", type=float, default=1)
    parser.add_argument("--mz_bin_post", type=float, default=0.2)
    parser.add_argument("--rt_bin_post", type=float, default=20)
    parser.add_argument("--mz_bin", type=float, default=0.2)
    parser.add_argument("--rt_bin", type=float, default=20)
    parser.add_argument("--run_name", type=str, default="eco,sag,efa,kpn,blk,pool")
    parser.add_argument("--scaler", type=str, default="none")
    parser.add_argument("--combat_corr", type=int, default=0)
    parser.add_argument("--is_sparse", type=int, default=1)
    parser.add_argument("--k", type=str, default=-1, help="Number of features to keep")
    parser.add_argument("--save", type=int, default=1, help="Save images and csvs?")
    parser.add_argument("--experiment", type=str, default='new_old_data')
    parser.add_argument("--resources_path", type=str, default='../../../resources',
                        help="Path to input directory")
    parser.add_argument("--feature_selection", type=str, default='mutual_info_classif', help="")
    parser.add_argument("--feature_selection_threshold", type=float, default=0.,
                        help="Mutual Information classification cutoff")
    parser.add_argument("--spd", type=str, default="200")
    parser.add_argument("--n_cpus", type=int, default=-1)
    parser.add_argument("--find_peaks", type=int, default=0)
    parser.add_argument("--lowess", type=int, default=0)
    parser.add_argument("--align_peaks", type=int, default=0)
    parser.add_argument("--make_data", type=int, default=0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--groupkfold", type=int, default=0)

    args = parser.parse_args()
    args.combat_corr = 0  # TODO to remove

    args.k = int(args.k)
    if float(args.mz_bin) >= 1:
        args.mz_bin = int(float(args.mz_bin))
    else:
        args.mz_bin = float(args.mz_bin)

    if float(args.rt_bin) >= 1:
        args.rt_bin = int(float(args.rt_bin))
    else:
        args.rt_bin = float(args.rt_bin)

    if float(args.mz_bin_post) >= 1:
        args.mz_bin_post = int(float(args.mz_bin_post))
    else:
        args.mz_bin_post = float(args.mz_bin_post)

    if float(args.rt_bin_post) >= 1:
        args.rt_bin_post = int(float(args.rt_bin_post))
    else:
        args.rt_bin_post = float(args.rt_bin_post)

    if args.mz_bin_post < 1:
        args.mz_rounding = len(str(args.mz_bin_post).split('.')[-1]) + 1
    else:
        args.mz_rounding = 1

    if args.rt_bin_post < 1:
        args.rt_rounding = len(str(args.rt_bin_post).split('.')[-1]) + 1
    else:
        args.rt_rounding = 1

    if args.test_run:
        args.run_name = 'test'


    bins = {
        'mz_bin_post': args.mz_bin_post,
        'rt_bin_post': args.rt_bin_post,
        'mz_bin': args.mz_bin,
        'rt_bin': args.rt_bin,
        'mz_rounding': args.mz_rounding,
        'rt_rounding': args.rt_rounding,
        'mz_shift': args.shift,
        'rt_shift': args.shift
    }

    out_dest = f"{args.resources_path}/{args.experiment}/matrices"

    script_dir = os.path.dirname(os.path.realpath(__file__))
    dir_name = f"{script_dir}/{out_dest}/mz{args.mz_bin}/rt{args.rt_bin}/mzp{args.mz_bin_post}/" \
               f"rtp{args.rt_bin_post}/thr{args.threshold}/{args.spd}spd/ms1/combat{args.combat_corr}/" \
               f"shift{args.shift}/{args.scaler}/log{args.log2}/{args.feature_selection}/"
    
    batches = ['03-04-2024', '29-03-2024', '01-03-2024', '21-02-2024', '26-02-2024', '13-03-2024', '02-02-2024']
    # batches = ['03-04-2024', '29-03-2024']
    # batches = ['01-03-2024']
    dir_inputs = []
    for batch in os.listdir(f"{script_dir}/{args.resources_path}/{args.experiment}"):
        if batch not in batches:
            continue
            continue
        if 'matrices' == batch or 'mzdb' == batch or 'time.txt' == batch:
            continue
        input_dir = f"{args.resources_path}/{args.experiment}/{batch}/tsv"
        dir_inputs += [f"{script_dir}/{input_dir}/mz{args.mz_bin}/rt{args.rt_bin}/{args.spd}spd/ms1/all/"]

    args.run_name = f"{args.run_name}_{'-'.join(batches)}_gkf{args.groupkfold}_{args.n_splits}splits"
    matrix_filename = f'{dir_name}/{args.run_name}/data_matrix_tmp.pkl'
    columns_filename = f'{dir_name}/{args.run_name}/columns.pkl'
    labels_filename = f'{dir_name}/{args.run_name}/labels.pkl'
    bacteria_to_keep = None

    if len(args.run_name.split(',')) > 1:
        bacteria_to_keep = args.run_name.split(',')
    else:
        bacteria_to_keep = None
    if args.make_data or not os.path.exists(f'{dir_name}/{args.run_name}'):

        data_matrices, labels, max_mzs, max_rts = [], [], [], []
        for dir_input in dir_inputs:
            tmp = make_df(dir_input, dir_name, bins=bins, args_dict=args, names_to_keep=bacteria_to_keep)
            data_matrices += [tmp[0]]
            labels += [tmp[1]]
            max_mzs += [tmp[3]['max_mz']]
            max_rts += [tmp[3]['max_rt']]

        labels = np.concatenate(labels)
        max_features = {'max_rt': max(max_rts), 'max_mz': max(max_mzs)}
        rts = [np.round(rt * args.rt_bin_post, args.rt_rounding) for rt in range(max_features['max_rt'])]
        mzs = [np.round(mz * args.mz_bin_post, args.mz_rounding) for mz in range(max_features['max_mz'])]
        columns = [f"{rt}_{mz}" for rt in rts for mz in mzs]

        data_matrices = adjust_tensors(data_matrices, max_features, args)
        data_matrix = vstack([vstack(data_matrices[k]) for k in range(len(data_matrices))])
        print('\nComplete data shape', data_matrix.shape)

        # Save the data matrix
        # Removes all the columns that are only zeros. Runs in parallel (only with 10% of the cpus)
        if args.align_peaks:
            print("Aligning the data...")
            peaks_list = pd.read_csv(f"{dir_name}/{args.run_name}/variance_scores.csv", index_col=0)
            data_matrix = msalign(data_matrix.columns, data_matrix.values)
        os.makedirs(f'{dir_name}/{args.run_name}', exist_ok=True)
        dump(data_matrix, open(matrix_filename, 'wb'))
        dump(columns, open(columns_filename, 'wb'))
        dump(labels, open(labels_filename, 'wb'))
    else:
        data_matrix = load(open(matrix_filename, 'rb'))
        columns = load(open(columns_filename, 'rb'))
        labels = load(open(labels_filename, 'rb'))
    print("Finding not zeros only columns...")
    print('\nComplete data shape', data_matrix.shape)

    # TODO only pass and return a list of list of columns. data_matrix could be seperated without taking more space
    # TODO now the space doubles here
    dframe_list = split_sparse(data_matrix, cols_per_split=int(1e5), columns=columns)

    n_cpus = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(int(n_cpus), maxtasksperchild=1)

    try:
        assert len(dframe_list[0]) == len(dframe_list[1])
    except AssertionError:
        print(len(dframe_list[0]), len(dframe_list[1]))
        exit('Columns and dataframes are not the same length') 

    fun = MultiKeepNotFunctionsSparse(keep_only_not_zeros_sparse, data=dframe_list[0], cols=dframe_list[1], nums=dframe_list[2],
                                      threshold=0, n_processes=np.ceil(data_matrix.shape[1] / int(1e5)))
    notzeros = pool.map(fun.process, range(len(dframe_list[0])))

    new_columns = np.array([x for x in np.concatenate([x[0] for x in notzeros])])
    not_zeros_col = np.array([x for x in np.concatenate([x[1] for x in notzeros])])
    # not_zeros_col = np.array([x for x in np.concatenate([x[1] for x in notzeros])])
    data_matrix = data_matrix[:, not_zeros_col]

    pool.close()
    pool.join()

    print("Finding not zeros columns...")
    print('\nComplete data shape', data_matrix.shape)
    dframe_list = split_sparse(data_matrix, cols_per_split=int(1e4), columns=new_columns)
    n_cpus = multiprocessing.cpu_count() - 1
    # if n_cpus > len(dframe_list):
    #     n_cpus = len(dframe_list)
    pool = multiprocessing.Pool(int(n_cpus), maxtasksperchild=1)
    fun = MultiKeepNotFunctionsSparse(keep_not_zeros_sparse, data=dframe_list[0], cols=dframe_list[1], nums=dframe_list[2],
                                       threshold=args.threshold, n_processes=np.ceil(data_matrix.shape[1] / int(1e4)))
    notzeros = pool.map(fun.process, range(len(dframe_list[0])))
    new_columns = np.array([x for x in np.concatenate([x[0] for x in notzeros])])
    not_zeros_col = np.array([x for x in np.concatenate([x[1] for x in notzeros])])
    data_matrix = data_matrix[:, not_zeros_col]

    pool.close()
    pool.join()

    columns = new_columns
    if args.log2 == 'after':
        print("Logging the data...")
        data_matrix = np.log1p(data_matrix)  # .astype(np.float32)
    if args.scaler in ['robust', 'standard', 'minmax']:
        if args.scaler == 'robust':
            print('Robust Standardization...')
            scaler = RobustScaler()
        elif args.scaler == 'standard':
            print('Standardization...')
            scaler = StandardScaler()
        elif args.scaler == 'minmax':
            print('Normalization...')
            scaler = MinMaxScaler()
        else:
            exit("Scaler not in ['robust', 'standard', 'minmax']")
        # labels = data_matrix.index
        columns = data_matrix.columns
        data_matrix = scaler.fit_transform(data_matrix)
        dump(scaler, open(f'{dir_name}/standard_scaler.pkl', 'wb'))
    elif args.scaler == 'none':
        print('No Normalization.')
    else:
        print('Scaler must be one of robust, standard, minmax or none')

    # The p (for plate) is removed to conform with the machine learning in the next step
    # labels = np.array(['_'.join(label.split('_p')) for label in labels])
    # data_matrix = pd.DataFrame(data_matrix.values, index=labels, columns=columns)
    # labels = data_matrix.index
    lows = []
    cats = []
    batches = []
    pool_indices = {'indices': [], 'names': []}
    for i, sample_name in enumerate(labels):
        # fname = file.split('\\')[-1]
        cat = sample_name.split('_')[0]
        batch = sample_name.split('_')[1]
        # batch = "_".join([sample_name.split('_')[1], sample_name.split('_')[2]])
        cats += [cat]
        batches += [batch]
        if 'pool' in sample_name:
            pool_indices['indices'] += [i]
            pool_indices['names'] += [labels[i]]
            lows += [0]
        elif 'l' in cat and 'blk' not in sample_name:
            lows += [1]
        else:
            lows += [0]

    pool_data = data_matrix[pool_indices['indices']]
    pool_batches = np.array(batches)[pool_indices['indices']].tolist()
    pool_labels = np.array(labels)[pool_indices['indices']].tolist()
    pool_cats = np.array(cats)[pool_indices['indices']].tolist()

    data = delete_rows_csr(data_matrix.tocsr(), pool_indices['names']).tocsc()
    cats = np.delete(np.array(cats), pool_indices['indices']).tolist()
    labels = np.delete(np.array(labels), pool_indices['indices']).tolist()
    lows = np.delete(np.array(lows), pool_indices['indices']).tolist()
    batches = np.delete(np.array(batches), pool_indices['indices']).tolist()

    _ = count_array(cats)
    cats = np.array(cats)

    if bacteria_to_keep is not None:
        bacteria_to_keep = [x if 'blk' not in x else 'blk' for x in bacteria_to_keep]

    print('\nComplete data shape', data.shape)

    fs = get_feature_selection_method(args.feature_selection)

    print(f"Calculating {args.feature_selection}\n")
    if args.feature_selection == 'variance':
        process_sparse_data(data, cats, columns, model=fs, dirname=dir_name, args=args)
    else:
        process_sparse_data_supervised(data, cats, batches, columns, model=fs, dirname=dir_name, args=args)

    args.mutual_info_path = f'{dir_name}/{args.run_name}/{args.feature_selection}_scores.csv'
    if args.k > -1:
        features = pd.read_csv(args.mutual_info_path)['minp_maxp_rt_mz'].to_numpy()[:args.k]
    else:
        features = pd.read_csv(args.mutual_info_path)['minp_maxp_rt_mz'].to_numpy()

    feats_pos = [np.argwhere(x==columns)[0][0] for x in features if x in columns]
    data = data[:, feats_pos]
    pool_data = pool_data[:, feats_pos]

    # Round values to 2 decimals
    data = np.round(np.nan_to_num(data), 2)
    pool_data = np.round(np.nan_to_num(pool_data), 2)

    # Make dataframes for BERNN
    # First column: sample IDs
    # Second column: labels
    # Third column: batch IDs
    # Rest of the columns: features
    # data = pd.concat([pd.DataFrame(data.index), pd.DataFrame(labels),
    #                              pd.DataFrame(batches), data], axis=1)
    infos = pd.concat([pd.DataFrame(cats), pd.DataFrame(batches)], axis=1)
    infos.index = labels
    infos.columns = ['label', 'batch']
    data = pd.concat([infos, pd.DataFrame(data.toarray(), index=labels, columns=features)], axis=1)
    pool_infos = pd.concat([pd.DataFrame(pool_cats), pd.DataFrame(pool_batches)], axis=1)
    pool_infos.index = pool_labels
    if len(pool_infos.index) > 0:
        pool_infos.columns = ['label', 'batch']
    pool_data = pd.concat([pool_infos, pd.DataFrame(pool_data.toarray(), index=pool_labels, columns=features)], axis=1)

    data.to_csv(
        f'{dir_name}/{args.run_name}/inputs.csv',
        index=True, index_label='ID')
    pool_data.to_csv(
        f'{dir_name}/{args.run_name}/pool_inputs.csv',
        index=True, index_label='ID')
    print('Duration: {}'.format(datetime.now() - start_time))

