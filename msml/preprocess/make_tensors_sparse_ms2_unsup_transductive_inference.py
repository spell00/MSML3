#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Simon Pelletier

"""

from datetime import datetime

start_time = datetime.now()
# import re
import os
import time
import csv
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
from features_selection_sparse import count_array, make_lists
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
from msalign import msalign
from functools import reduce
from scipy import sparse
# import copyfile
from shutil import copyfile
from utils import adjust_tensors
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(filename='make_tensors_ms2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

GLOBAL_TIMES = csv.writer(open('global_times.csv', 'w'))


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

        # Starts a csv file to record the time it takes to process each file
        os.makedirs(f'{path}/{args.run_name}', exist_ok=True)
        with open(f'{path}/{args.run_name}/time.csv', 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['File', 'Time', 'input_size(MB)', 'output_size(MB)'])

        os.makedirs(f'{path}/{args.run_name}/images', exist_ok=True)
        os.makedirs(f'{path}/{args.run_name}/csv', exist_ok=True)

        self.bins = bins
        self.is_sparse = args.is_sparse

        self.path = path
        # path to the tsv files
        self.tsv_path = path.split('matrices')[0]
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
        self.args = args
        self.run_path = '\\'.join(self.tsv_list[0].split('/')[:-1])
        # reinitiale the time file
        with open(f'{self.run_path}/time.csv', 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['File', 'Time', 'input_size(MB)', 'output_size(MB)'])
        

    def process(self, index):
        """

        This process makes a 'tensor' (list of dataframes)
        :param i:
        :return:
        """
        try:
            startTime = time.time()
            # print(len(gc.get_objects()))
            try:
                file, label = self.tsv_list[index], self.labels_list[index]
            except:
                exit('Error with tsv index')
            try:
                tsv = pd.read_csv(file, header=0, sep='\t')
                tsv = tsv.astype({c: np.float32 for c in tsv.select_dtypes(include='float64').columns})
                tsv_size = tsv.memory_usage().sum() / 2 ** 20
            except:
                exit('Error reading csv')
            print(
                f"Processing file {index}: {file} min_parents: min={tsv.min_parent_mz.min()} max={tsv.min_parent_mz.max()}")

            # if label != 'aba_01-03-2024_240222_u002_l':
            #     return None, None, None
            tsv = tsv[tsv.bin_intensity != 0]

            tsv = tsv.drop(['max_parent_mz'], axis=1)
            # tsv['mz_bin'] = tsv['mz_bin'].round(2)

            min_parents_mz = np.unique(tsv.min_parent_mz)
            # Find all intervals between min_parents. Then get mode interval
            intervals = np.diff(min_parents_mz)
            if len(np.unique(intervals)) > 1:
                print(f"MUTLIPLE Intervals between min_parents: {np.unique(intervals)}")
            interval = np.round(stats.mode(intervals)[0], 2)
            # Remove from intervals values that are not at an interval distance of the next value
            min_parents_mz = np.array([x for x in min_parents_mz if x + interval in min_parents_mz or x - interval in min_parents_mz])
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

            try:
                final = {min_parent: pd.DataFrame(
                    np.zeros([int(np.ceil(tsv.mz_bin.max() / self.bins['mz_bin_post'])) + 1,
                            int(np.ceil(tsv.rt_bin.max() / self.bins['rt_bin_post'])) + 1]),
                    dtype=np.float32,
                    columns=np.arange(0, tsv.rt_bin.max() + self.bins['rt_bin_post'], self.bins['rt_bin_post']).round(
                        self.bins['rt_rounding']) - rt_shift,
                    index=np.arange(0, tsv.mz_bin.max() + self.bins['mz_bin_post'], self.bins['mz_bin_post']).round(
                        self.bins['mz_rounding']) - mz_shift
                ) for min_parent in
                        np.arange(int(min_parents_mz.min()), int(min_parents_mz.max()) + interval, interval)}
            except:
                exit('Error creating final dataframe')
            for i in list(final.keys()):
                final[i].index = np.round(final[i].index, self.bins['mz_rounding'])
                final[i].columns = np.round(final[i].columns, self.bins['rt_rounding'])
            min_parent = list(final.keys())[0]
            rt = float(final[min_parent][list(final[min_parent].keys())[0]][0])
            # spdtypes = final[min_parent].dtypes[rt]
            # prev_mz = -1
            try:
                missed = 0
                for i, line in enumerate(tsv.values):
                    min_parent, rt, mz, intensity = line
                    if np.isnan(rt) or np.isnan(mz) or np.isnan(min_parent):
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
                    # final[min_parent][mz][rt] += np.log1p(intensity)
                    mz = np.round(float(mz), self.bins['mz_rounding'])
                    rt = np.round(float(rt), self.bins['rt_rounding'])
                    # if self.is_sparse and prev_mz != rt:
                        # Change todense on mz rather than rt
                        # if prev_mz != -1:
                        #     final[min_parent] = final[min_parent].astype(spdtypes)
                        # final[min_parent] = final[min_parent].sparse.to_dense()
                    try:
                        if self.log2 == 'inloop':
                            final[min_parent].loc[mz].loc[rt] += np.log1p(intensity)
                        else:
                            final[min_parent].loc[mz].loc[rt] += intensity
                    except:
                        missed += 1
                    # if self.is_sparse:
                    #     final[min_parent][rt] = final[min_parent][rt].astype(spdtypes)
                    # del min_parent, rt, mz, intensity, line
                    # prev_mz = mz
                    if self.test_run and i > 10000:
                        break
            except:
                exit('Error filling final dataframe')
            if missed > 0:
                print(f"Missed {missed} values in file {index}")
            for min_parent in final:
                final[min_parent] = final[min_parent].astype(dtype)
            if self.lowess:
                for df in final:
                    for ii, line1 in enumerate(final[df].values):
                        final[df].iloc[ii] = lowess(line1, list(final[df].columns), frac=0.1, return_sorted=False)
            if self.find_peaks:
                for df in final:
                    for ii, line1 in enumerate(final[df].values):
                        mask = find_peaks(final[df].iloc[ii], height=0.1, distance=2)
                        # Make all values 0 except the ones that are in mask
                        final[df].iloc[ii] = pd.Series([final[df].iloc[ii].values[i] if i in mask[0] else 0 for i in range(len(final[df].iloc[ii]))])
            # os.makedirs(f"{self.path}/nibabel/", exist_ok=True)
            if self.save:
                # img = nib.Nifti1Image(np.stack(list(final.values())), np.eye(4))
                # img.uncache()
                # nib.save(img, f'{self.path}/nibabel/{label}.nii')
                df = np.stack(list(final.values()))
                # df = df / df.max(axis=(1, 2), keepdims=True)
                df = df / df.max()
                _ = [self.save_images_and_csv3d(matrix, df[i], f"{label}", min_parent) for i, (min_parent, matrix) in
                    enumerate(zip(final, list(final.values())))]
                
                df = df.sum(0)
                df = df / df.max()
                _ = self.save_images_and_csv(reduce(lambda x, y: x.astype(float).add(y.astype(float)), final.values()), df, label)
                # del img
            # df = np.stack(list(final.values()))
            # for x in list(final.keys()):
            #     final[x] = csc_matrix(final[x])
            total_memory = np.sum([final[x].memory_usage().sum() for x in list(final.keys())]) / 2 ** 20
            total_time = (time.time() - startTime)
            # Round to 2 decimals
            total_memory = np.round(total_memory, 2)
            total_time = np.round(total_time, 2)
            tsv_size = np.round(tsv_size, 2)
            print(
                f"Finished file {index}. Total memory: {total_memory}  MB, time: {total_time} seconds"
            )
            csv_file = f'{self.run_path}/time.csv'
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([label, total_time, tsv_size, total_memory])
        except Exception as e:
            print(f"Error with file {index}: {file}")
            print(e)
            return None, None, None
        return final, list(final.keys()), label

    def n_samples(self):
        """
        Gets n samples. Mainly just to have a second class so pylint does not complain.
        """
        return len(self.tsv_list)

    def save_images_and_csv3d(self, final, df, label, i):
        os.makedirs(f"{self.path}/{self.args.exp_name}/csv3d/{label}/", exist_ok=True)
        os.makedirs(f"{self.path}/{self.args.exp_name}/images3d/{label}/", exist_ok=True)
        final.to_csv(f"{self.path}/{self.args.exp_name}/csv3d/{label}/{label}_{i}.csv", index_label='ID')
        im = Image.fromarray(np.uint8(cm.gist_earth(df) * 255))
        im.save(f"{self.path}/{self.args.exp_name}/images3d/{label}/{label}_{i}.png")
        im.close()
        del im

    def save_images_and_csv(self, final, df, label):
        os.makedirs(f"{self.path}/{self.args.exp_name}/csv/", exist_ok=True)
        os.makedirs(f"{self.path}/{self.args.exp_name}/images/", exist_ok=True)
        final.to_csv(f"{self.path}/{self.args.exp_name}/csv/{label}.csv", index_label='ID')
        im = Image.fromarray(np.uint8(cm.gist_earth(df) * 255))
        im.save(f"{self.path}/{self.args.exp_name}/images/{label}.png")
        im.close()
        del im

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

    parents = np.unique(np.concatenate([np.array(x[1]) for x in data_matrix]))
    diff_parents = int(parents[1] - parents[0])
    n_min_mz_parent = max([len(x[0]) for x in data_matrix])
    max_rt = max([max([x[0][y].shape[1] for y in x[0]]) for x in data_matrix])
    max_mz = max([max([x[0][y].shape[0] for y in x[0]]) for x in data_matrix])

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
            if n_min_mz_parent - len(matrix) < 0:
                logging.warning(
                    f'mz{args_dict.mz_bin} rt{args_dict.rt_bin} mzp{args_dict.mz_bin_post} rtp{args_dict.rt_bin_post} : {label} had different number of min_mz_parent')
                data_matrix = data_matrix[1:]
                labels = labels[1:]
                pbar.update(1)
                continue
            elif n_min_mz_parent - len(matrix) > 0:
                # find the missing min_mz_parent
                not_in_keys = [x for x in parents if x not in matrix.keys()]
                for x in not_in_keys:
                    matrix = {**matrix, x: csc_matrix(np.zeros((max_mz, max_rt)))}

            # https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
            # hstack of csc matrices should be faster than coo (worst) or csr
            for x in list(matrix.keys()):
                if max_mz - matrix[x].shape[0] > 0:
                    matrix[x] = csc_matrix(
                        vstack((
                            matrix[x], np.zeros((int((max_mz - matrix[x].shape[0])), matrix[x].shape[1])))
                        ))
                else:
                    matrix[x] = csc_matrix(matrix[x])
                if max_rt - matrix[x].shape[1] > 0:
                    matrix[x] = csc_matrix(
                        hstack((
                            matrix[x], csc_matrix(np.zeros((matrix[x].shape[0], int((max_rt - matrix[x].shape[1])))))
                        ))
                    )
                else:
                    matrix[x] = csc_matrix(matrix[x])

            matrices += [matrix]
            new_labels += [label]
            data_matrix = data_matrix[1:]
            labels = labels[1:]
            del matrix
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
    mz_min_parents = np.arange(min(parents), max(parents) + diff_parents, diff_parents)

    return matrices, new_labels, [f"{mz_min_parent}_{rt}_{mz}" for mz_min_parent in mz_min_parents for rt in rts for mz
                                   in mzs], {'parents': parents, 'max_rt': max_rt, 'max_mz': max_mz}
   
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
    parser.add_argument("--min_mz", type=int, default=100)
    parser.add_argument("--max_mz", type=int, default=1200)
    parser.add_argument("--min_rt", type=int, default=140)
    parser.add_argument("--max_rt", type=int, default=320)
    parser.add_argument("--train_batches", type=str, default='B14-B13-B12-B11-B10-B9-B8-B7-B6-B5-B4-B3-B2-B1')
    parser.add_argument("--decimals", type=int, default=-1)

    args = parser.parse_args()
    args.exp_name = f'all_{args.train_batches}_gkf{args.groupkfold}_mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}_{args.n_splits}splits'

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

    # Directory where the matrices will be saved
    out_dest = f"{args.resources_path}/{args.experiment}/matrices"

    # Directory of the present script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Directory of the inputs
    dir_name = f"{script_dir}/{out_dest}/mz{args.mz_bin}/rt{args.rt_bin}/mzp{args.mz_bin_post}/" \
               f"rtp{args.rt_bin_post}/thr{args.threshold}/{args.spd}spd/ms2/combat{args.combat_corr}/" \
               f"shift{args.shift}/{args.scaler}/log{args.log2}/{args.feature_selection}/"
    
    batches = ["B15-06-29-2024"]
    dir_inputs = []
    for batch in batches:
        if batch not in batches:
            continue
        if 'matrices' == batch or 'mzdb' == batch or 'time.txt' == batch:
            continue
        input_dir = f"{args.resources_path}/{args.experiment}/{batch}/tsv"
        dir_inputs += [f"{script_dir}/{input_dir}/mz{args.mz_bin}/rt{args.rt_bin}/{args.spd}spd/ms2/all/"]

    # We might not need to keep all the signals
    cropings = f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}"

    # The name of the run is the name of the run + the name of the batches
    args.run_name = f"{args.run_name}_{'-'.join([b.split('-')[0] for b in batches])}_gkf{args.groupkfold}_{cropings}_{args.n_splits}splits"
    matrix_filename = f'{dir_name}/{args.run_name}/data_matrix_tmp.pkl'
    columns_filename = f'{dir_name}/{args.exp_name}/columns_nozeros.pkl'
    labels_filename = f'{dir_name}/{args.run_name}/labels.pkl'
    bacteria_to_keep = None

    # if columns_filename not exists leave with message that the training has not been done
    if not os.path.exists(columns_filename):
        print(f"Columns file {columns_filename} does not exist. Run the training first.")
        exit()

    if len(args.run_name.split(',')) > 1:
        bacteria_to_keep = args.run_name.split(',')
    else:
        bacteria_to_keep = None
    if args.make_data or not os.path.exists(f'{dir_name}/{args.run_name}'):
        data_matrices, labels, max_mzs, max_rts, parents = [], [], [], [], []
        for dir_input in dir_inputs:
            print(f"Processing {dir_input}")
            # Start a timer to record the time it takes to process one df
            timer_df = time.time()
            tmp = make_df(dir_input, dir_name, bins=bins, args_dict=args, names_to_keep=bacteria_to_keep)
            data_matrices += [tmp[0]]
            labels += [tmp[1]]
            max_mzs += [tmp[3]['max_mz']]
            max_rts += [tmp[3]['max_rt']]
            parents += [tmp[3]['parents']]
            # Calculate time for make_df in seconds
            timer_df = time.time() - timer_df
            GLOBAL_TIMES.writerow([dir_input, timer_df])

        # Adjustements timer
        timer_adustments = time.time()
        labels = np.concatenate(labels)
        parents = np.unique(np.concatenate(parents))
        max_features = {
            'parents': len(parents),
            'max_rt': max(max_rts), 
            'max_mz': max(max_mzs)
        }
        # Creation of the columns
        rts = [np.round(rt * args.rt_bin_post, args.rt_rounding) for rt in range(max_features['max_rt'])]
        mzs = [np.round(mz * args.mz_bin_post, args.mz_rounding) for mz in range(max_features['max_mz'])]
        columns = np.array([f"{mz_min_parent}_{rt}_{mz}" for mz_min_parent in parents for rt in rts for mz in mzs])

        # Adjust the tensors to have the same number of features
        data_matrices = adjust_tensors(data_matrices, max_features, args)

        # When all tensors have the same number of features, we can stack them (sparse matrices)
        data_matrix = vstack([vstack(data_matrices[k]) for k in range(len(data_matrices))])
        print('\nComplete data shape', data_matrix.shape)

        # Save the data matrix
        # Removes all the columns that are only zeros. Runs in parallel (only with 10% of the cpus)
        if args.align_peaks:
            print("Aligning the data...")
            peaks_list = pd.read_csv(f"{dir_name}/{args.run_name}/variance_scores.csv", index_col=0)
            data_matrix = msalign(data_matrix.columns, data_matrix.values)
        os.makedirs(f'{dir_name}/{args.run_name}', exist_ok=True)
        columns_to_keep = load(open(columns_filename, 'rb'))
        to_keep = np.array([i for i, x in enumerate(columns) if x in columns_to_keep])  # TODO THIS IS CORRECT. THE ERROR MUST BE FURTHER DOWN
        data_matrix = sparse.lil_matrix(data_matrix[:,to_keep])  # TODO There are other instances where I could have used this function
        dump(data_matrix, open(matrix_filename, 'wb'))
        dump(labels, open(labels_filename, 'wb'))
    else:
        data_matrix = load(open(matrix_filename, 'rb'))
        columns_to_keep = columns = load(open(columns_filename, 'rb'))
        labels = load(open(labels_filename, 'rb'))
    
    print('\nComplete data shape', data_matrix.shape)
    # Timer
    timer_adustments = time.time() - timer_adustments
    GLOBAL_TIMES.writerow(['Adjustments', timer_adustments])
    # columns = new_columns
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
        cat = sample_name.split('_')[0]
        batch = sample_name.split('_')[1]
        cats += [cat]
        batches += [batch.split('-')[0]]
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

    # Time reordering of features
    ordering_timer = time.time()
    args.mutual_info_path = f'{dir_name}/{args.exp_name}/{args.feature_selection}_scores.csv'
    if args.feature_selection != 'none':
        if args.k > -1:
            features = pd.read_csv(args.mutual_info_path)['minp_maxp_rt_mz'].to_numpy()[:args.k]
        else:
            features = pd.read_csv(args.mutual_info_path)['minp_maxp_rt_mz'].to_numpy()
    else:
        features = columns_to_keep
    copyfile(args.mutual_info_path, f'{dir_name}/{args.run_name}/{args.feature_selection}_scores.csv')
    # Copy columns_filename from exp_name to run_name
    copyfile(columns_filename, f'{dir_name}/{args.run_name}/columns_nozeros.pkl')  # TODO MAYBE THE ERROR IS HERE??

    feats_pos = [np.argwhere(x==columns_to_keep)[0][0] for x in features if x in columns_to_keep]
    # Save the feats_pos if file not exist yet, else load it
    if not os.path.exists(f'{dir_name}/{args.exp_name}/feats_pos.pkl'):
        dump(feats_pos, open(f'{dir_name}/{args.run_name}/feats_pos.pkl', 'wb'))
    else:
        feats_pos = load(open(f'{dir_name}/{args.run_name}/feats_pos.pkl', 'rb'))
    # Find all the columns that are not in the features
    feats_absent = [x for x in columns if x not in features]
    # Make an array of zeros to add to the data
    zeros = csc_matrix(np.zeros((data.shape[0], len(feats_absent))))
    # Add the zeros to the data
    data = hstack([data, zeros])
    pool_data = hstack([pool_data, zeros])
    # Add the absent features to the columns
    data = data.todense()[:, feats_pos]
    pool_data = pool_data.todense()[:, feats_pos]

    # Round values to 2 decimals
    if args.decimals > 0:
        data = np.round(np.nan_to_num(data), args.decimals)
        try:
            pool_data = np.round(np.nan_to_num(pool_data), args.decimals)
        except:
            pass
    ordering_timer = time.time() - ordering_timer
    GLOBAL_TIMES.writerow(['Reordering', ordering_timer])
    # Timer to save files
    save_time = time.time()
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
    data = pd.concat([infos, pd.DataFrame(data, index=labels, columns=features)], axis=1)
    pool_infos = pd.concat([pd.DataFrame(pool_cats), pd.DataFrame(pool_batches)], axis=1)
    pool_infos.index = pool_labels
    if len(pool_infos.index) > 0:
        pool_infos.columns = ['label', 'batch']
        pool_data = pd.concat([pool_infos, pd.DataFrame(pool_data, index=pool_labels, columns=features)], axis=1)
        pool_data.to_csv(
            f'{dir_name}/{args.run_name}/pool_inputs_{args.feature_selection}.csv',
            index=True, index_label='ID')

    batches = '-'.join([b.split('-')[0] for b in np.unique(batches)])
    data.to_csv(
        f'{dir_name}/{args.run_name}/inputs_{args.feature_selection}.csv',
        index=True, index_label='ID'
    )
    save_time = time.time() - save_time
    GLOBAL_TIMES.writerow(['Save files', save_time])
    print('Total Duration: {}'.format(datetime.now() - start_time))
    GLOBAL_TIMES.writerow(['Total Duration', datetime.now() - start_time])

