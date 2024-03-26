#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHUL of Québec, Canada
Author: Simon Pelletier
June 2021

"""

from datetime import datetime

start_time = datetime.now()
import os
import glob
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif  # , RFE, RFECV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from PIL import Image
from matplotlib import cm
from pickle import dump
import nibabel as nib
from sklearn.model_selection import StratifiedKFold
from msml.utils.batch_effect_removal import remove_batch_effect_all, get_berm
from msml.utils.features_selection import get_feature_selection_method, keep_only_not_zeros, keep_not_zeros, \
    process_data, count_array, make_lists, split_df, MultiKeepNotFunctions


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

def get_data_matrix(args):
    """
    Function that takes the command line arguments (args) and gets the appropriate data.
    The main thing the function does is to prepare the arguments to be passed to the function
    `make_df`. If using "shifted" data, it calls `make_df` multiple times and creates a larger
    dataframe.
    Args:
        args:

    Returns:
        data_matrix
    """
    bins = {
        'mz_bin_post': args.mz_bin_post,
        'rt_bin_post': args.rt_bin_post,
        'mz_rounding': args.mz_rounding,
        'rt_rounding': args.rt_rounding,
        'mz_shift': args.shift,
        'rt_shift': args.shift
    }

    out_dest = f"{args.resources_path}/{args.experiment}/matrices"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dir_name = f"{script_dir}/{out_dest}/mz{args.mz_bin}/rt{args.rt_bin}/mzp{args.mz_bin_post}/" \
               f"rtp{args.rt_bin_post}/{args.spd}spd/ms1/combat{args.combat_corr}/" \
               f"shift{args.shift}/{args.scaler}/log{args.log2}/{args.feature_selection}/"
    dir_inputs = []
    for batch in os.listdir(f"{script_dir}/{args.resources_path}/{args.experiment}"):
        input_dir = f"{args.resources_path}/{args.experiment}/{batch}/tsv"
        dir_inputs += [f"{script_dir}/{input_dir}/mz{args.mz_bin}/rt{args.rt_bin}/{args.spd}spd/ms1/all/"]

    bacteria_to_keep = None

    if len(args.run_name.split(',')) > 1:
        bacteria_to_keep = args.run_name.split(',')
    else:
        bacteria_to_keep = None
    data_matrices = []
    for dir_input in dir_inputs:
        data_matrices += [make_df(dir_input, dir_name, bins=bins, args_dict=args,
                                        names_to_keep=bacteria_to_keep)]
        # if args.shift:
        #     data_matrix2 = make_df(dir_input, dir_name, bins=bins, args_dict=args,
        #                                         names_to_keep=bacteria_to_keep)
        #     data_matrix3 = make_df(dir_input, dir_name, bins=bins, args_dict=args,
        #                                         names_to_keep=bacteria_to_keep)
        #     data_matrix4 = make_df(dir_input, dir_name, bins=bins, args_dict=args,
        #                                         names_to_keep=bacteria_to_keep)
        #     data_matrix = pd.concat((data_matrix, data_matrix2, data_matrix3, data_matrix4), 1)
    # data_matrix.to_csv('')
    # data_matrix = data_matrix.view(data_matrix.shape[0], -1)
    data_matrix = pd.concat(data_matrices, 0)

    return data_matrix, dir_name


class MakeTensorsMultiprocess:
    """
    Class to read the data contained in tsv files and make
     binned matrices returned as pandas dataframes
    """

    def __init__(self, tsv_list, labels_list, bins, test_run, n_samples, log, path, save):
        """
        When initiated, the object needs a few parameters
         that are shared by all parallel processes.
        :param tsv_list:
        :param labels_list:
        :param test_run:
        :return:
        """
        os.makedirs(f'{path}/images', exist_ok=True)
        os.makedirs(f'{path}/csv', exist_ok=True)

        self.bins = bins

        self.path = path
        self.tsv_list = tsv_list
        self.labels_list = labels_list
        self.save = save
        self.test_run = test_run
        # self.mz_shift = mz_shift
        # self.rt_shift = rt_shift
        self.log2 = log
        # self.mz_rounding = mz_rounding
        # self.rt_rounding = rt_rounding
        # self.final_list = []
        # self.labels = []
        if n_samples != -1:
            self.tsv_list = self.tsv_list[:n_samples]
            self.labels_list = self.labels_list[:n_samples]

    def process(self, i):
        """
        This process makes a 'tensor' (list of dataframes)
        :param i:
        :return:
        """
        file, label = self.tsv_list[i], self.labels_list[i]
        try:
            tsv = pd.read_csv(file, header=0, sep='\t', dtype=np.float64)
        except:
            exit('Error reading csv')
        print(f"Processing file {i}")

        tsv = tsv[tsv.bin_intensity != 0]

        tsv['mz_bin'] = tsv['mz_bin'].round(2)

        if self.bins['mz_shift']:
            mz_shift = self.bins['mz_bin_post'] / 2
        else:
            mz_shift = 0
        if self.bins['rt_shift']:
            rt_shift = self.bins['rt_bin_post'] / 2
        else:
            rt_shift = 0

        try:
            final_df = pd.DataFrame(
                np.zeros([int(np.ceil(tsv.rt_bin.max() / self.bins['rt_bin_post'])) + 1,
                          int(np.ceil(tsv.mz_bin.max() / self.bins['mz_bin_post'])) + 1]),
                dtype=np.float64,
                index=np.arange(0, tsv.rt_bin.max() + self.bins['rt_bin_post'], self.bins['rt_bin_post']).round(
                    self.bins['rt_rounding']) - rt_shift,
                columns=np.arange(0, tsv.mz_bin.max() + self.bins['mz_bin_post'], self.bins['mz_bin_post']).round(
                    self.bins['mz_rounding']) - mz_shift
            )
        except:
            final_df = pd.DataFrame([])

        final_df.index = np.round(final_df.index, self.bins['rt_rounding'])
        final_df.columns = np.round(final_df.columns, self.bins['mz_rounding'])

        for i, line in enumerate(tsv.values):
            try:
                rt, mz, intensity = line
            except:
                pass
            if np.isnan(rt) or np.isnan(mz):
                continue
            if self.bins['rt_shift']:
                tmp_rt = np.floor(np.round(rt / self.bins['rt_bin_post'], 8)) * self.bins['rt_bin_post']
                if rt % (self.bins['rt_bin_post'] / 2) > (self.bins['rt_bin_post'] / 2):
                    tmp_rt += self.bins['rt_bin_post'] / 2
                else:
                    tmp_rt -= self.bins['rt_bin_post'] / 2
                rt = tmp_rt
                # if not self.mz_shift:
                # mz = np.floor(np.round(mz/mz_bin, 8))*mz_bin
            else:
                rt = np.floor(np.round(rt / self.bins['rt_bin_post'], 8)) * self.bins['rt_bin_post']

            if self.bins['mz_shift']:
                tmp_mz = np.floor(np.round(mz / self.bins['mz_bin_post'], 8)) * self.bins['mz_bin_post']
                if mz % (self.bins['mz_bin_post'] / 2) > (self.bins['mz_bin_post'] / 2):
                    tmp_mz += self.bins['mz_bin_post'] / 2
                else:
                    tmp_mz -= self.bins['mz_bin_post'] / 2
                mz = tmp_mz
                # if not self.rt_shift:
                # rt = np.floor(np.round(rt/rt_bin, 8))*rt_bin
            else:
                # rt = np.floor(np.round(rt/rt_bin, 8))*rt_bin
                mz = np.floor(np.round(mz / self.bins['mz_bin_post'], 8)) * self.bins['mz_bin_post']
            if self.bins['rt_rounding'] != 0:
                rt = np.round(rt, self.bins['rt_rounding'])
            if self.bins['mz_rounding'] != 0:
                mz = np.round(mz, self.bins['mz_rounding'])

            if self.log2 == 'inloop':
                final_df[mz][rt] += np.log1p(intensity)
            else:
                final_df[mz][rt] += intensity
            if self.test_run and i == 10:
                break
        os.makedirs(f"{self.path}/nibabel/", exist_ok=True)
        img = nib.Nifti1Image(final_df.values, np.eye(4))
        nib.save(img, f'{self.path}/nibabel/{label}.nii')
        if self.save:
            try:
                _ = self.save_images_and_csv(final_df, f"{label}")
            except:
                pass

        return final_df, label

    def n_samples(self):
        """
        Gets n samples. Mainly just to have a second class so pylint does not complain.
        """
        return len(self.tsv_list)

    def save_images_and_csv(self, final, label):
        os.makedirs(f"{self.path}/csv/", exist_ok=True)
        os.makedirs(f"{self.path}/images/", exist_ok=True)
        final.to_csv(f"{self.path}/csv/{label}.csv", index_label='ID')
        im = Image.fromarray(np.uint8(cm.gist_earth(final.values) * 255))
        im.save(f"{self.path}/images/{label}.png")
        im.close()


# def make_lists(dirinput, path, run_name):
#     """
#     Makes lists
# 
#     :param dirinput:
#     :param path:
#     :param run_name:
#     :return:
#     """
#     tsv_list = glob.glob(dirinput + '*.tsv')
#     # Initiate variables
#     samples = []
#     # pool_files = []
# 
#     # for _, file in enumerate(tsv_list):
#     #     if 'pool' in file:
#     #         pool_files += [file]
# 
#     # for psample in pool_files:
#     #     tsv_list.remove(psample)
#     labels_list = []
#     for _, file in enumerate(tsv_list):
#         if len(file.split('\\')) > 1:
#             sample = file.split('\\')[-1].split('.')[0]
#         else:
#             sample = file.split('/')[-1].split('.')[0]
# 
#         samples.append(sample)
#         tmp = sample.split('_')
#         batch = tmp[3]
#         label = tmp[1]
#         concentration = tmp[-2]
#         urine = tmp[-3]
# 
#         # label = f"{batch}_{'_'.join(tmp[-3:])}".lower()
#         label = f"{batch}_{label}_{concentration}_{urine}".lower()
#         labels_list.append(label)
# 
#     categories = [x.split('_')[0] for x in labels_list]
# 
#     names_df = pd.DataFrame(
#         np.concatenate(
#             (np.array(labels_list).reshape((-1, 1)),
#              np.array(samples).reshape((-1, 1)),
#              np.array(categories).reshape((-1, 1)),
#              ), 1)
#     )
#     os.makedirs(path, exist_ok=True)
#     names_df.to_csv(f'{path}/fnames_ids_{run_name}.csv', index=False,
#                     header=['ID', 'fname', 'category'])
#     return {
#         "samples": samples,
#         "tsv": tsv_list,
#         "labels": labels_list,
#     }
# 

def make_df(dirinput, dirname, bins, args_dict, names_to_keep=None, features=None):
    """
    Concatenate and transpose

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

    concat = MakeTensorsMultiprocess(lists["tsv"], lists["labels"], bins,
                                     args_dict.test_run, args_dict.n_samples,
                                     args_dict.log2, dirname, args_dict.save)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    data_matrix = pool.map(concat.process,
                           range(len(concat.tsv_list))
                           )
    print('Tensors are done.')
    labels = [x[1] for x in data_matrix]
    # Find max of each dimension
    if features is not None:
        max_rt1 = int(np.ceil(max([float(x.split('_')[0]) for x in features]) / float(args_dict.rt_bin_post)))
        max_mz1 = int(np.ceil(max([float(x.split('_')[1]) for x in features]) / float(args_dict.mz_bin_post)))
        max_rt2 = max([x[0].shape[1] for x in data_matrix])
        max_mz2 = max([x[0].shape[2] for x in data_matrix])
        max_rt = int(max(max_rt1, max_rt2))
        max_mz = int(max(max_mz1, max_mz2))
    else:
        max_rt = max([x[0].shape[0] for x in data_matrix])
        max_mz = max([x[0].shape[1] for x in data_matrix])

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
    for matrix, label in zip([x[0].values for x in data_matrix], labels):
        if max_rt - matrix.shape[0] > 0:
            matrix = np.concatenate(
                (matrix, np.zeros((int((max_rt - matrix.shape[0])), matrix.shape[1]))), 0)
        if max_mz - matrix.shape[1] > 0:
            matrix = np.concatenate(
                (matrix, np.zeros((matrix.shape[0], int((max_mz - matrix.shape[1]))))), 1)
        matrices += [matrix.reshape(-1)]
        new_labels += [label]
    pool.close()
    pool.join()
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

    return pd.DataFrame(
        data=np.stack(matrices),
        index=new_labels,
        columns=[f"{rt}_{mz}" for rt in rts for mz in mzs]
    ).fillna(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--test_run", type=int, default=0,
                        help="Is it a test run? 1 for true 0 for false")
    parser.add_argument("--n_samples", type=int, default=-1,
                        help="How many samples to run? Only modify to make test runs faster (-1 for all samples)")
    parser.add_argument("--log2", type=str, default='inloop',
                        help='log the data in the loop, after the loop or not log the data. Choices: [inloop, after, no]')
    parser.add_argument("--shift", type=int, default=0, help='Shift the data matrix')
    parser.add_argument("--binary", type=int, default=0, help='Blanks vs bacteria')
    parser.add_argument("--threshold", type=float, default=0.9)
    # parser.add_argument("--mz_rounding", type=int, default=1)
    # parser.add_argument("--rt_rounding", type=int, default=1)
    parser.add_argument("--mz_bin_post", type=float, default=0.2)
    parser.add_argument("--rt_bin_post", type=float, default=20)
    parser.add_argument("--mz_bin", type=float, default=0.2)
    parser.add_argument("--rt_bin", type=float, default=20)
    parser.add_argument("--run_name", type=str, default="eco,sag,efa,kpn,blk,pool", help='To select specific ')
    parser.add_argument("--scaler", type=str, default="none")
    parser.add_argument("--combat_corr", type=int, default=0)
    parser.add_argument("--use_test", type=int, default=0)
    parser.add_argument("--use_valid", type=int, default=0)
    parser.add_argument("--k", type=str, default=-1, help="Number of features to keep")
    parser.add_argument("--save", type=int, default=1, help="Save images and csvs?")
    parser.add_argument("--resources_path", type=str, default='../../../../resources',
                        help="Path to input directory")
    parser.add_argument("--experiment", type=str, default='20220706_Data_ML02/Data_FS')

    parser.add_argument("--feature_selection", type=str, default='mutual_info_classif',
                        help="Mutual Information classification cutoff")
    parser.add_argument("--feature_selection_threshold", type=float, default=0.,
                        help="Mutual Information classification cutoff")
    parser.add_argument("--spd", type=str, default="200")
    parser.add_argument('--batch_removal_method', type=str, default='none')
    args = parser.parse_args()
    args.combat_corr = 0  # TODO to remove

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
    data_matrix, dir_name = get_data_matrix(args)
    if args.test_run:
        args.run_name = 'test'
    print('\nComplete data shape', data_matrix.shape)

    # Removes all the columns that are only zeros. Runs in parallel (only with 10% of the cpus)
    print("Finding not zeros columns...")
    if not args.test_run:
        # DataFrame is split into smaller dataframes to enable parallel processing
        dframe_list = split_df(data_matrix, cols_per_split=int(1e5))
    else:
        dframe_list = split_df(data_matrix.iloc[:, :1000], cols_per_split=int(1e2))

    n_cpus = multiprocessing.cpu_count()
    if n_cpus > len(dframe_list):
        n_cpus = len(dframe_list)
    pool = multiprocessing.Pool(int(n_cpus))

    fun = MultiKeepNotFunctions(keep_only_not_zeros, dframe_list, threshold=0,
                                n_processes=np.ceil(data_matrix.shape[1] / int(1e5)))
    data_matrix = pool.map(fun.process, range(len(dframe_list)))
    data_matrix, not_zeros_col = pd.concat([x[0] for x in data_matrix], 1), np.array(
        [x for x in np.concatenate([x[1] for x in data_matrix])])
    pool.close()
    pool.join()

    print("Finding not zeros columns...")
    dframe_list = split_df(data_matrix, cols_per_split=int(1e4))
    n_cpus = multiprocessing.cpu_count()
    if n_cpus > len(dframe_list):
        n_cpus = len(dframe_list)
    pool = multiprocessing.Pool(int(n_cpus))
    fun = MultiKeepNotFunctions(keep_not_zeros, dframe_list, threshold=args.threshold,
                                n_processes=np.ceil(data_matrix.shape[1] / int(1e4)))
    data_matrix = pool.map(fun.process, range(len(dframe_list)))
    data_matrix, not_zeros_col = pd.concat([x[0] for x in data_matrix], 1), np.array(
        [x for x in np.concatenate([x[1] for x in data_matrix])])
    pool.close()
    pool.join()

    # not_zeros_col = np.unique(np.concatenate((not_zeros_columns, not_zeros_columns2)))
    labels = data_matrix.index
    columns = data_matrix.columns
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
        labels = data_matrix.index  # TODO Remove this
        columns = data_matrix.columns
        data_matrix = scaler.fit_transform(data_matrix)
        dump(scaler, open(f'{dir_name}/standard_scaler.pkl', 'wb'))
    elif args.scaler == 'none':
        print('No Normalization.')
    else:
        print('Scaler must be one of robust, standard, minmax or none')
    # labels = np.array(['_'.join(label.split('_p')) for label in labels])

    data_matrix = pd.DataFrame(data_matrix.values, index=labels, columns=columns)
    labels = data_matrix.index  # TODO remove this
    lows = []
    cats = []
    batches = []
    pool_indices = {'indices': [], 'names': []}
    for i, sample_name in enumerate(labels):
        # fname = file.split('\\')[-1]
        cat = sample_name.split('_')[1]
        batch = sample_name.split('_')[0]
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

    pool_data = data_matrix.iloc[pool_indices['indices']]
    pool_batches = np.array(batches)[pool_indices['indices']].tolist()
    pool_labels = np.array(labels)[pool_indices['indices']].tolist()

    train_data = data_matrix.drop(pool_indices['names'])
    cats = np.delete(np.array(cats), pool_indices['indices']).tolist()
    labels = np.delete(np.array(labels), pool_indices['indices']).tolist()
    lows = np.delete(np.array(lows), pool_indices['indices']).tolist()
    batches = np.delete(np.array(batches), pool_indices['indices']).tolist()

    train_data = remove_batch_effect_all(get_berm(args.batch_removal_method), train_data, all_batches=batches)

    _ = count_array(cats)
    cats = np.array(cats)

    final = {
        'train': train_data.copy(),
        'test': None,
        'valid': None,
        'pool': pool_data.copy()
    }

    cats = {
        'train': cats,
        'valid': None,
        'test': None
    }

    print("\n\nTests\n\n")

    # The blks in tests are all in the same plate

    skf = StratifiedKFold(n_splits=5)
    train_nums = np.arange(0, len(data_matrix))
    train_inds, test_inds = skf.split(train_nums, cats['train']).__next__()
    data_matrix, test_data_matrix = data_matrix.iloc[train_inds], data_matrix.iloc[test_inds]
    cats['train'] = cats['train'][train_inds]
    final['train'] = final['train'].iloc[train_inds]

    print('\nComplete test data shape', test_data_matrix.shape)

    cols = [True if x in test_data_matrix.columns else False for i, x in enumerate(not_zeros_col)]
    test_data_matrix = test_data_matrix[not_zeros_col[cols]]
    data_matrix = data_matrix[not_zeros_col[cols]]

    print('Standardization...')
    test_labels = test_data_matrix.index
    columns = test_data_matrix.columns
    if args.log2 == 'after':
        test_data_matrix = np.log1p(test_data_matrix)  # .astype(np.float32)
    if args.scaler != 'none':
        test_data_matrix = scaler.transform(test_data_matrix)

    print("Logging the data...")
    test_data_matrix = pd.DataFrame(test_data_matrix, index=test_labels, columns=columns)
    test_labels = test_data_matrix.index
    test_lows = []
    test_cats = []
    test_batches = []
    test_pool_indices = {'indices': [], 'names': []}
    for i, sample_name in enumerate(test_labels):
        # fname = file.split('\\')[-1]
        cat = sample_name.split('_')[1]
        batch = sample_name.split('_')[0]

        test_cats += [cat]
        test_batches += [0]
        # test_batches += [batch]
        if 'pool' in sample_name:
            test_pool_indices['indices'] += [i]
            test_pool_indices['names'] += [test_labels[i]]
            test_lows += [0]
        elif 'l' in cat and 'blk' not in sample_name:
            test_lows += [1]
        else:
            test_lows += [0]

    test_pool_data = test_data_matrix.iloc[test_pool_indices['indices']]
    test_pool_batches = np.array(test_batches)[test_pool_indices['indices']].tolist()

    test_final = test_data_matrix.drop(test_pool_indices['names'])
    test_cats = np.delete(np.array(test_cats), test_pool_indices['indices']).tolist()
    test_labels = np.delete(np.array(test_labels), test_pool_indices['indices']).tolist()
    test_lows = np.delete(np.array(test_lows), test_pool_indices['indices']).tolist()
    test_batches = np.delete(np.array(test_batches), test_pool_indices['indices']).tolist()

    _ = count_array(test_cats)

    test_cats = np.array(test_cats)

    test_final.columns = final['train'].columns
    test_pool_data.columns = final['train'].columns
    final['test'] = test_final.copy()
    final['test_pool'] = test_pool_data.copy()

    print("\n\nValidations\n\n")

    # The blks in tests are all in the same plate
    skf = StratifiedKFold(n_splits=5)
    train_nums = np.arange(0, len(data_matrix))
    train_inds, valid_inds = skf.split(train_nums, cats['train']).__next__()
    data_matrix, valid_data_matrix = data_matrix.iloc[train_inds], data_matrix.iloc[valid_inds]
    cats['train'] = cats['train'][train_inds]
    final['train'] = final['train'].iloc[train_inds]
    print('\nComplete valid data shape', data_matrix.shape)

    cols = [True if x in valid_data_matrix.columns else False for i, x in enumerate(not_zeros_col)]

    # TODO solve this Problem: Duplicated column names as mz_bin_post=1!!
    valid_data_matrix = valid_data_matrix[not_zeros_col[cols]]
    data_matrix = data_matrix[not_zeros_col[cols]]

    print('Standardization...')
    valid_labels = valid_data_matrix.index
    columns = valid_data_matrix.columns
    if args.log2 == 'after':
        valid_data_matrix = np.log1p(valid_data_matrix)  # .astype(np.float32)
    if args.scaler != 'none':
        valid_data_matrix = scaler.transform(valid_data_matrix)

    print("Logging the data...")
    valid_data_matrix = pd.DataFrame(valid_data_matrix, index=valid_labels, columns=columns)
    valid_labels = valid_data_matrix.index
    valid_lows = []
    valid_cats = []
    valid_batches = []
    valid_pool_indices = {'indices': [], 'names': []}
    for i, sample_name in enumerate(valid_labels):
        # fname = file.split('\\')[-1]
        cat = sample_name.split('_')[1]
        batch = sample_name.split('_')[0]

        valid_cats += [cat]
        valid_batches += [0]
        # valid_batches += [batch]
        if 'pool' in sample_name:
            valid_pool_indices['indices'] += [i]
            valid_pool_indices['names'] += [valid_labels[i]]
            valid_lows += [0]
        elif 'l' in cat and 'blk' not in sample_name:
            valid_lows += [1]
        else:
            valid_lows += [0]

    valid_pool_data = valid_data_matrix.iloc[valid_pool_indices['indices']]
    valid_pool_batches = np.array(valid_batches)[valid_pool_indices['indices']].tolist()

    valid_final = valid_data_matrix.drop(valid_pool_indices['names'])
    valid_cats = np.delete(np.array(valid_cats), valid_pool_indices['indices']).tolist()
    valid_labels = np.delete(np.array(valid_labels), valid_pool_indices['indices']).tolist()
    valid_lows = np.delete(np.array(valid_lows), valid_pool_indices['indices']).tolist()
    valid_batches = np.delete(np.array(valid_batches), valid_pool_indices['indices']).tolist()

    _ = count_array(valid_cats)

    valid_cats = np.array(valid_cats)

    valid_final.columns = final['train'].columns
    valid_pool_data.columns = final['train'].columns
    final['valid'] = valid_final.copy()
    final['valid_pool'] = valid_pool_data.copy()

    print('\nComplete data shape', final['train'].shape)
    print('\nComplete valid data shape', final['valid'].shape)
    print('\nComplete test data shape', final['test'].shape)

    fs = get_feature_selection_method(args.feature_selection)

    print(f"Calculating {args.feature_selection}\n")
    process_data(final, cats, model=fs, cutoff=args.feature_selection_threshold,
                 dirname=dir_name, k=int(args.k), feature_selection=args.feature_selection, run_name=args.run_name,
                 combat_corr=args.combat_corr)

    mutual_info_path = f'{dir_name}/{args.run_name}/{args.feature_selection}_scores.csv'
    if args.k > -1:
        features = pd.read_csv(mutual_info_path)['minp_maxp_rt_mz'].to_numpy()[:args.k]
    else:
        features = pd.read_csv(mutual_info_path)['minp_maxp_rt_mz'].to_numpy()

    # The uncombatted values are also saved
    test_final = test_final[features]
    test_pool_data = test_pool_data[features]
    valid_final = valid_final[features]
    valid_pool_data = valid_pool_data[features]
    train_data = train_data[features]
    pool_data = pool_data[features]
    final['valid'] = final['valid'][features]
    final['valid_pool'] = final['valid_pool'][features]
    final['test'] = final['test'][features]
    final['test_pool'] = final['test_pool'][features]

    # Round values to 2 decimals
    final['test'].iloc[:] = np.round(np.nan_to_num(final['test']), 2)
    final['test_pool'].iloc[:] = np.round(np.nan_to_num(final['test_pool']), 2)
    final['valid'].iloc[:] = np.round(np.nan_to_num(final['valid']), 2)
    final['valid_pool'].iloc[:] = np.round(np.nan_to_num(final['valid_pool']), 2)
    valid_final.iloc[:] = np.round(np.nan_to_num(valid_final), 2)
    valid_pool_data.iloc[:] = np.round(np.nan_to_num(valid_pool_data), 2)
    test_final.iloc[:] = np.round(np.nan_to_num(test_final), 2)
    test_pool_data.iloc[:] = np.round(np.nan_to_num(test_pool_data), 2)
    train_data.iloc[:] = np.round(np.nan_to_num(train_data), 2)
    pool_data.iloc[:] = np.round(np.nan_to_num(pool_data), 2)

    if args.combat_corr:
        final['test'].to_csv(
            f'{dir_name}/{args.run_name}/test_inputs_combat.csv',
            index=True, index_label='ID')
        final['test_pool'].to_csv(
            f'{dir_name}/{args.run_name}/test_pool_inputs_combat.csv',
            index=True, index_label='ID')
        final['valid'].to_csv(
            f'{dir_name}/{args.run_name}/valid_inputs_combat.csv',
            index=True, index_label='ID')
        final['valid_pool'].to_csv(
            f'{dir_name}/{args.run_name}/valid_pool_inputs_combat.csv',
            index=True, index_label='ID')
        test_final.to_csv(
            f'{dir_name}/{args.run_name}/test_inputs.csv',
            index=True, index_label='ID')
        test_pool_data.to_csv(
            f'{dir_name}/{args.run_name}/test_pool_inputs.csv',
            index=True, index_label='ID')
        valid_final.to_csv(
            f'{dir_name}/{args.run_name}/valid_inputs.csv',
            index=True, index_label='ID')
        valid_pool_data.to_csv(
            f'{dir_name}/{args.run_name}/valid_pool_inputs.csv',
            index=True, index_label='ID')

        train_data.to_csv(
            f'{dir_name}/{args.run_name}/train_inputs.csv',
            index=True, index_label='ID')
        pool_data.to_csv(
            f'{dir_name}/{args.run_name}/train_pool_inputs.csv',
            index=True, index_label='ID')
    else:
        final['test'].to_csv(
            f'{dir_name}/{args.run_name}/test_inputs.csv',
            index=True, index_label='ID')
        final['test_pool'].to_csv(
            f'{dir_name}/{args.run_name}/test_pool_inputs.csv',
            index=True, index_label='ID')
        final['valid'].to_csv(
            f'{dir_name}/{args.run_name}/valid_inputs.csv',
            index=True, index_label='ID')
        final['valid_pool'].to_csv(
            f'{dir_name}/{args.run_name}/valid_pool_inputs.csv',
            index=True, index_label='ID')
    print('Duration tsv2df: {}'.format(datetime.now() - start_time))
