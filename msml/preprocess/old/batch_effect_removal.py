import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface import NULL
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()
robust_scaler = RobustScaler()

base = importr('base')
stats = importr('stats')


def comBatR(data, batches, orders=None,classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values.T)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    sva = importr('sva')
    stats = importr('stats')

    if classes != NULL:
        mod = stats.model_matrix(~base.as_factor(classes))
    newdata = sva.ComBat(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata.T


def harmonyR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    harmony = importr('harmony')
    newdata = harmony.HarmonyMatrix(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def waveicaR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    waveica = importr('WaveICA')
    data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    newdata = waveica.WaveICA(dat=data_r, batch=batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def seuratR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    waveica = importr('WaveICA')
    data_r.colnames = robjects.StrVector([str(x) for x in range(df.shape[1])])
    newdata = waveica.WaveICA(dat=data_r, batch=batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def qcrlscR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    """
    Batches need to be qc or not qc
    Args:
        data:
        batches:
        orders:
        classes:
        par_prior:
        ref_batch:

    Returns:

    """
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    rcpm = importr('Rcpm')
    if orders is None:
        orders = robjects.IntVector(list(range(batches.reshape(-1).shape[0])))
    newdata = rcpm.qc_rlsc(data_r, batches_r, orders)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def fasticaR(data, batches, orders=None, classes=NULL, par_prior=True, ref_batch=NULL):
    """
    Batches need to be qc or not qc
    Args:
        data:
        batches:
        orders:
        classes:
        par_prior:
        ref_batch:

    Returns:

    """
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    ica = importr('ica')
    newdata = ica.icafast(data_r, df.shape[0])
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def zinbWaveR(data, batches, orders=None,classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    zinbwave = importr('zinbwave')
    granges = importr('GenomicRanges')
    sumexp = importr('SummarizedExperiment')
    iranges = importr('IRanges')
    s4vectors = importr('S4Vectors')

    nrows = 200
    ncols = 6
    counts = robjects.r.matrix(stats.runif(nrows * ncols, 1, 1e4), nrows)
    rowRanges = granges.GRanges(base.rep(base.c("chr1", "chr2"), base.c(50, 150)),
                                iranges.IRanges(base.floor(stats.runif(200, 1e5, 1e6)), width=100),
                                strand=base.sample(base.c("+", "-"), 200, True),
                                )
    colData = s4vectors.DataFrame(Treatment=base.rep(base.c("ChIP", "Input"), 3), row_names=base.LETTERS[0:6])

    exp = sumexp.SummarizedExperiment(assays=base.list(counts=counts),
                                      rowRanges=rowRanges, colData=colData)

    data_assay_r = sumexp.SummarizedExperiment(data_r)
    newdata = zinbwave.zinbwave(data_assay_r, K=2, epsilon=1000)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def ligerR(data, batches, orders=None,classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)
    df = df.iloc[:1000]
    data_r = robjects.r.array(robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0]))
    batches_r = robjects.IntVector(batches.reshape(-1))
    rliger = importr('rliger')
    newdata = rliger.normalize(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def scMergeR(data, batches, orders=None,classes=NULL, par_prior=True, ref_batch=NULL):
    df = pd.DataFrame(data.values)

    data_r = robjects.r.matrix(robjects.FloatVector(df.values.reshape(-1)), nrow=df.shape[0])
    batches_r = robjects.IntVector(batches.reshape(-1))
    scMerge = importr('scMerge')
    newdata = scMerge.scMerge(data_r, batches_r)
    with localconverter(robjects.default_converter + pandas2ri.converter):
        newdata = np.array(robjects.conversion.rpy2py(newdata))
    return newdata


def remove_batch_effect(berm, all_data, train_data, valid_data, test_data, train_pool_data, valid_pool_data, test_pool_data, all_batches, orders=None):
    """
    All dataframes have a shape of N samples (rows) x M features (columns)

    Args:
        berm: Batch effect removal method
        all_data:
        train_data:
        valid_data:
        test_data:
        all_batches:
        orders:

    Returns:
        Returns:
        A dictionary of pandas datasets with keys:
            'all': Pandas dataframe containing all data (train, valid and test data),
            'train': Pandas dataframe containing the training data,
            'valid': Pandas dataframe containing the validation data,
            'test: Pandas dataframe containing the test data'

    """
    if berm is not None:
        df = pd.DataFrame(all_data)
        # df[df.isna()] = 0
        all_data = berm(df, all_batches, orders)
        all_data = pd.DataFrame(all_data, index=df.index, columns=df.columns)
        train_data = all_data.iloc[:train_data.shape[0]]
        valid_data = all_data.iloc[train_data.shape[0]:train_data.shape[0] + valid_data.shape[0]]
        test_data = all_data.iloc[train_data.shape[0] + valid_data.shape[0]:train_data.shape[0] + valid_data.shape[0] + test_data.shape[0]]
        train_pool_data = all_data.iloc[train_data.shape[0] + valid_data.shape[0] + test_data.shape[0]:train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] + train_pool_data.shape[0]]
        valid_pool_data = all_data.iloc[train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] + train_pool_data.shape[0]:train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] + train_pool_data.shape[0] + valid_pool_data.shape[0]]
        test_pool_data = all_data.iloc[train_data.shape[0] + valid_data.shape[0] + test_data.shape[0] + train_pool_data.shape[0] + valid_pool_data.shape[0]:]

    return {'all': all_data, 'train': train_data, 'valid': valid_data, 'test': test_data, 'train_pool': train_pool_data, 'valid_pool': valid_pool_data, 'test_pool': test_pool_data}


def remove_batch_effect_all(berm, all_data, all_batches):
    """
    All dataframes have a shape of N samples (rows) x M features (columns)

    Args:
        berm: Batch effect removal method
        all_data:
        all_batches:

    Returns:
        Returns:
            Pandas dataframe containing all data (train, valid and test data),

    """
    if berm is not None:
        df = pd.DataFrame(all_data)
        # df[df.isna()] = 0
        all_data = berm(df, all_batches)

    return all_data


def get_berm(berm):
    # berm: batch effect removal method
    if berm == 'combat':
        berm = comBatR
    if berm == 'harmony':
        berm = harmonyR
    if berm == 'waveica':
        berm = waveicaR
    if berm == 'qcrlsc':
        berm = qcrlscR
    if berm == 'ica':
        berm = fasticaR
    if berm == 'none':
        berm = None
    return berm

