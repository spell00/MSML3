import numpy as np
import pandas as pd
import csv
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

def rLISI(data, meta_data, perplexity=10):
    lisi = importr('lisi')
    # all_batches_r = robjects.IntVector(all_batches[all_ranks])
    # all_data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    # labels = ['label1', 'label2']
    labels = robjects.StrVector(['label1'])
    new_meta_data = robjects.r.matrix(robjects.IntVector(meta_data), nrow=data.shape[0])
    newdata = robjects.r.matrix(robjects.FloatVector(data.values.reshape(-1)), nrow=data.shape[0])

    new_meta_data.colnames = labels
    results = lisi.compute_lisi(newdata, new_meta_data, labels, perplexity)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results = np.array(robjects.conversion.rpy2py(results))
    mean = np.mean(results)
    return mean  # , np.std(results), results

def rKBET(inputs, cats):
    kbet = importr('kBET')
    # all_batches_r = robjects.IntVector(all_batches[all_ranks])
    # all_data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    # labels = ['label1', 'label2']
    labels = robjects.StrVector(['label1'])
    new_meta_data = robjects.IntVector(cats)
    newdata = robjects.r.matrix(robjects.FloatVector(inputs.values.reshape(-1)), nrow=inputs.shape[0])

    new_meta_data.colnames = labels
    results = kbet.kBET(newdata, new_meta_data, do_pca=False, plot=False)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        results = robjects.conversion.rpy2py(results[0])
    try:
        mean = results['kBET.signif'][0]
    except:
        mean = 0

    return mean

def read_csv(csv_file, num_rows=1000, n_cols=1000):
    # data = np.array([])
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        if n_cols != -1:
            progress_bar = tqdm(csv_reader, desc="Reading CSV")
            n_cols = n_cols + 3
        else:
            progress_bar = tqdm(csv_reader, desc="Reading CSV")
        for row_num, row in enumerate(csv_reader):
            row = np.array(row)[:n_cols]
            if num_rows != -1:
                if row_num >= num_rows:
                    break
            # if len(data) == 0:
            #     data = row.reshape(1, -1)
            # else:
            #     data = np.concatenate((data, row.reshape(1, -1)), 0)
            data += [row]
            progress_bar.update(1)
            del row
    data = np.stack(data)
    return pd.DataFrame(data[1:, :], columns=data[0, :])

