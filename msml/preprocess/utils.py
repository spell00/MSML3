# import os
import logging
import numpy as np
import pandas as pd
# import multiprocessing
from tqdm import tqdm
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix, vstack, hstack, csc_matrix




def crop_data(data, columns, args):
    """
    Crop the data to the min and max mz and rt values
    :param data:
    :param columns:
    :param args:
    :return:
    """
    mz_min = args.min_mz
    mz_max = args.max_mz
    rt_min = args.min_rt
    rt_max = args.max_rt
    rts = np.array([float(x.split("_")[1]) for x in columns])
    mzs = np.array([float(x.split("_")[2]) for x in columns])
    rts_to_keep = (rts >= rt_min) & (rts <= rt_max)
    mzs_to_keep = (mzs >= mz_min) & (mzs <= mz_max)

    data = data[:, rts_to_keep & mzs_to_keep]
    columns = columns[rts_to_keep & mzs_to_keep]
    return data, columns


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
    n_parents = max_features['parents']
    max_rt = max_features['max_rt']
    max_mz = max_features['max_mz']
    # TODO could be parallelized if worth it
    for j, matrices in enumerate(list_matrices):
        with tqdm(total=len(matrices), position=0, leave=True) as pbar:
            for i, matrix in enumerate(matrices):
                # matrix = data_matrix[0][0]
                if n_parents - len(matrix) > 0:
                    logging.warning(
                        f'mz{args_dict.mz_bin} rt{args_dict.rt_bin} mzp{args_dict.mz_bin_post} rtp{args_dict.rt_bin_post} : {label} had different number of min_mz_parent')
                    pbar.update(1)
                    continue

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
                matrices[i] = hstack([matrix[x].reshape(1, -1) for x in list(matrix.keys())]).tocsr()
                del matrix
                pbar.update(1)
        list_matrices[j] = matrices
    print('Tensors are adjusted.')
    return list_matrices


