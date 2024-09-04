import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
import csv
from tqdm import tqdm
from msml.utils.utils import get_unique_labels
from scipy.sparse import vstack, csr_matrix
import os

def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

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
            if row_num == 0:
                if num_rows == -1:
                    num_rows = sum(1 for _ in open(csv_file, 'rb'))
                else:
                    num_rows = min(num_rows, sum(1 for _ in open(csv_file, 'rb')))
                data_num = np.empty((num_rows-1, len(row)-3))
                data_str = np.empty((num_rows-1, 3), dtype=object)
                header = row
                continue
            if n_cols != -1:
                row = np.array(row)[:n_cols]
            else:
                row = np.array(row)
            if num_rows != -1:
                if row_num >= num_rows:
                    break

            data_num[row_num-1] = row[3:].astype(float)
            data_str[row_num-1] = row[:3]
            progress_bar.update(1)
            del row

    data = np.concatenate((data_str, data_num), 1)

    return pd.DataFrame(data, columns=header)

def read_csv_low_ram(csv_file, num_rows=1000, n_cols=1000):
    data = None
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        if n_cols != -1:
            progress_bar = tqdm(csv_reader, desc="Reading CSV")
            n_cols = n_cols + 3
        else:
            progress_bar = tqdm(csv_reader, desc="Reading CSV")
        for row_num, row in enumerate(csv_reader):
            if n_cols != -1:
                row = np.array(row)[:n_cols]
            else:
                row = np.array(row)
            if num_rows != -1:
                if row_num >= num_rows:
                    break
            if data is None:
                data = row.reshape(1, -1)
            else:
                data = np.concatenate((data, row.reshape(1, -1)), 0)
            progress_bar.update(1)
            del row
            # print(row.shape)
    # data = np.stack(data)
    return pd.DataFrame(data[1:, :], columns=data[0, :])



def get_data_infer(path, args, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        data
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'names', 'labels', 'cats', 'batches', 'manips', 'orders', 'sets', 'urines', 'concs']:
        data[info] = {}
    data[info]['all'] = data[info]['test'] = data[info]['urinespositives'] = np.array([])
    matrix = read_csv(csv_file=f"{path}/{args.csv_file}", num_rows=-1, n_cols=args.n_features)
    top_features = pd.read_csv(f"{path}/{args.features_file}")
    names = matrix.iloc[:, 0]
    labels = matrix.iloc[:, 1]

    if args.binary:
        labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in labels])

    batches = matrix.iloc[:, 2]
    manips = pd.Series([x.split("_")[2] for x in names])
    urines = pd.Series([x.split("_")[3] for x in names])
    concs = pd.Series([x.split("_")[4] if len(x.split("_")) == 5 else 'na' for x in names])
    unique_batches = batches.unique()
    unique_manips = manips.unique()
    unique_urines = urines.unique()
    unique_concs = concs.unique()
    # batches = np.stack([np.argwhere(x == unique_batches).squeeze() for x in batches])
    orders = np.array([0 for _ in batches])
    matrix = matrix.iloc[:, 3:].fillna(0).astype(float)
    # Array of colnames that are not in matrix.columns
    cols = [x for x in top_features.iloc[:, 0].values if x not in matrix.columns]
    # Add zero columns if they are not in matrix.columns
    if len(cols) > 0:
        matrix = pd.concat((matrix, pd.DataFrame(np.zeros((matrix.shape[0], len(cols))), columns=cols)), 1)
    else:
        pass
    # top_features = top_features[top_features.iloc[:, 0].isin(matrix.columns)]
    matrix = matrix.loc[:, top_features.iloc[:, 0].values[:args.n_features]]
    if args.remove_zeros:
        mask1 = (matrix == 0).mean(axis=0) < 0.1
        matrix = matrix.loc[:, mask1]
    if args.log1p:
        matrix.iloc[:] = np.log1p(matrix.values)
    matrix.iloc[:] = np.nan_to_num(matrix.values)

    columns = matrix.columns
    mz_parents = [float(column.split('_')[0]) for column in columns]
    mzs = [float(column.split('_')[2]) for column in columns]
    rts = [float(column.split('_')[1]) for column in columns]
    columns_to_keep = [True if (mzp >= args.min_mz_parent and mzp <= args.max_mz_parent) \
                        and (mz >= args.min_mz and mz <= args.max_mz) \
                        and (rt >= args.min_rt and rt <= args.max_rt) \
                        else False for name, mzp, mz, rt in zip(columns, mz_parents, mzs, rts)
                        ]

    matrix = matrix.loc[:, columns_to_keep]
    # pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
    pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' not in name]

    pos = [i for i, name in enumerate(names.values.flatten()) if 'urinespositives' in name]
    not_pos = [i for i, name in enumerate(names.values.flatten()) if 'urinespositives' not in name]
    data['inputs']["urinespositives"], data['inputs']['test'] = matrix.iloc[pos], matrix.iloc[not_pos]
    data['names']["urinespositives"], data['names']['test'] = names.to_numpy()[pos], names.to_numpy()[not_pos]
    data['labels']["urinespositives"], data['labels']['test'] = labels.to_numpy()[pos], labels.to_numpy()[not_pos]
    data['batches']["urinespositives"], data['batches']['test'] = batches.to_numpy()[pos], batches.to_numpy()[not_pos]
    data['manips']["urinespositives"], data['manips']['test'] = manips.to_numpy()[pos], manips.to_numpy()[not_pos]
    data['urines']["urinespositives"], data['urines']['test'] = urines.to_numpy()[pos], urines.to_numpy()[not_pos]
    data['concs']["urinespositives"], data['concs']['test'] = concs.to_numpy()[pos], concs.to_numpy()[not_pos]
    data['orders']["urinespositives"], data['orders']['test'] = orders[pos], orders[not_pos]

    unique_labels = np.array(np.unique(data['labels']['test']))
    # place blancs at the end
    blanc_class = np.argwhere(unique_labels == 'blanc').flatten()[0]
    unique_labels = np.concatenate((np.delete(unique_labels, blanc_class), ['blanc']))
    data['cats']['test'] = np.array(
        [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels']['test'])])

    if args.pool:
        pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
        data['inputs'][f"test_pool"] = matrix.iloc[pool_pos]
        data['names'][f"test_pool"] = np.array([f'pool_{i}' for i, _ in enumerate(pool_pos)])
        data['labels'][f"test_pool"] = np.array([f'pool' for _ in pool_pos])
        data['batches'][f"test_pool"] = batches[pool_pos]
        data['manips'][f"test_pool"] = manips[pool_pos]
        data['urines'][f"test_pool"] = urines[pool_pos]
        data['concs'][f"test_pool"] = concs[pool_pos]

        # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
        data['orders'][f"test_pool"] = orders[pool_pos]
        data['cats'][f"test_pool"] = np.array(
            [len(np.unique(data['labels']['test'])) for _ in batches[pool_pos]])

        data['labels']['test'] = np.array([x.split('-')[0] for i, x in enumerate(data['labels']['test'])])
        unique_labels = np.concatenate((unique_labels, np.array(['pool'])))
        data['cats']['test'] = np.array(
            [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels']['test'])])

    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
    if not args.pool:
        for key in list(data.keys()):
            if key in ['inputs']:
                data[key]['all'] = data[key]['test']
            else:
                data[key]['all'] = data[key]['test']
        
        unique_batches = np.unique(data['batches']['all'])
        # for group in ['train', 'valid', 'test', 'all']:
        #     data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])
    else:
        for key in list(data.keys()):
            if key in ['inputs']:
                data[key]['all'] = pd.concat((
                    data[key]['test'], data[key]['test_pool'],
                ), 0)
                data[key]['all_pool'] = data[key]['test_pool']
            else:
                data[key]['all'] = np.concatenate((
                    data[key]['test'], data[key]['test_pool']
                ), 0)
                data[key]['all_pool'] = data[key]['test_pool']

        unique_batches = np.unique(data['batches']['all'])
        # for group in ['train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all', 'all_pool']:
        #     data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches, unique_manips, unique_urines, unique_concs


def get_data(path, args, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        data
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'names', 'labels', 'cats', 'batches', 'manips', 'orders', 'sets', 'urines', 'concs']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid']:
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                splitter = skf.split(train_nums, data['labels']['train'], data['batches']['train'])

            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                splitter = skf.split(train_nums, data['labels']['train'])

            _, valid_inds = splitter.__next__()
            _, test_inds = splitter.__next__()
            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[valid_inds], data['inputs']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][valid_inds], data['labels']['train'][test_inds]
            data['names']['train'], data['names']['valid'], data['names']['test'] = data['names']['train'][train_inds], \
                data['names']['train'][valid_inds], data['names']['train'][test_inds]
            data['orders']['train'], data['orders']['valid'], data['orders']['test'] = data['orders']['train'][train_inds], \
                data['orders']['train'][valid_inds], data['orders']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'], data['batches']['test'] = data['batches']['train'][train_inds], \
                data['batches']['train'][valid_inds], data['batches']['train'][test_inds]
            data['manips']['train'], data['manips']['valid'], data['manips']['test'] = data['manips']['train'][train_inds], \
                data['manips']['train'][valid_inds], data['manips']['train'][test_inds]
            data['cats']['train'], data['cats']['valid'], data['cats']['test'] = data['cats']['train'][train_inds], data['cats']['train'][
                valid_inds], data['cats']['train'][test_inds]
            data['urines']['train'], data['urines']['valid'], data['urines']['test'] = data['urines']['train'][train_inds], \
                data['urines']['train'][valid_inds], data['urines']['train'][test_inds]
            data['concs']['train'], data['concs']['valid'], data['concs']['test'] = data['concs']['train'][train_inds], data['concs']['train'][
                valid_inds], data['concs']['train'][test_inds]

            if args.pool:
                if args.groupkfold:
                    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                    train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                    pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'],
                                                    data['batches']['train_pool'])

                else:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                    pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'])

                _, valid_inds = pool_splitter.__next__()
                _, test_inds = pool_splitter.__next__()
                train_inds = [x for x in train_nums_pool if x not in np.concatenate((valid_inds, test_inds))]
                data['inputs']['train_pool'], data['inputs']['valid_pool'], data['inputs']['test_pool'], = data['inputs']['train_pool'].iloc[train_inds], \
                    data['inputs']['train_pool'].iloc[valid_inds], data['inputs']['train_pool'].iloc[test_inds]
                data['labels']['train_pool'], data['labels']['valid_pool'], data['labels']['test_pool'], = data['labels']['train_pool'][train_inds], \
                    data['labels']['train_pool'][valid_inds], data['labels']['train_pool'][test_inds]
                data['names']['train_pool'], data['names']['valid_pool'], data['names']['test_pool'], = data['names']['train_pool'][train_inds], \
                    data['names']['train_pool'][valid_inds], data['names']['train_pool'][test_inds]
                data['orders']['train_pool'], data['orders']['valid_pool'], data['orders']['test_pool'], = data['orders']['train_pool'][train_inds], \
                    data['orders']['train_pool'][valid_inds], data['orders']['train_pool'][test_inds]
                data['batches']['train_pool'], data['batches']['valid_pool'], data['batches']['test_pool'], = data['batches']['train_pool'][train_inds], \
                    data['batches']['train_pool'][valid_inds], data['batches']['train_pool'][test_inds]
                data['manips']['train_pool'], data['manips']['valid_pool'], data['manips']['test_pool'], = data['manips']['train_pool'][train_inds], \
                    data['manips']['train_pool'][valid_inds], data['manips']['train_pool'][test_inds]
                data['cats']['train_pool'], data['cats']['valid_pool'], data['cats']['test_pool'], = data['cats']['train_pool'][train_inds], data['cats']['train_pool'][
                    valid_inds], data['cats']['train_pool'][test_inds]
                data['urines']['train_pool'], data['urines']['valid_pool'], data['urines']['test_pool'], = data['urines']['train_pool'][train_inds], \
                    data['urines']['train_pool'][valid_inds], data['urines']['train_pool'][test_inds]
                data['concs']['train_pool'], data['concs']['valid_pool'], data['concs']['test_pool'], = data['concs']['train_pool'][train_inds], data['concs']['train_pool'][
                    valid_inds], data['concs']['train_pool'][test_inds]

        else:
            if args.low_ram:
                matrix = read_csv_low_ram(csv_file=f"{path}/{args.csv_file}", num_rows=-1, n_cols=args.n_features)
            else:
                matrix = read_csv(csv_file=f"{path}/{args.csv_file}", num_rows=-1, n_cols=args.n_features)
            matrix.index = names = matrix.iloc[:, 0]
            names = matrix.iloc[:, 0]
            labels = matrix.iloc[:, 1]

            # if args.binary:
            #     labels = pd.Series(['blanc' if label=='blanc' else 'bact' for label in labels])

            batches = matrix.iloc[:, 2]
            manips = pd.Series([x.split("_")[2] for x in names])
            urines = pd.Series([x.split("_")[3] for x in names])
            concs = pd.Series([x.split("_")[4] if len(x.split("_")) == 5 else 'na' for x in names])
            unique_batches = batches.unique()
            unique_manips = manips.unique()
            unique_urines = urines.unique()
            unique_concs = concs.unique()
            # batches = np.stack([np.argwhere(x == unique_batches).squeeze() for x in batches])
            orders = np.array([0 for _ in batches])
            matrix = matrix.iloc[:, 3:].fillna(0).astype(float)

            columns = matrix.columns
            mz_parents = [float(column.split('_')[0]) for column in columns]
            mzs = [float(column.split('_')[2]) for column in columns]
            rts = [float(column.split('_')[1]) for column in columns]
            columns_to_keep = [True if (mzp >= args.min_mz_parent and mzp <= args.max_mz_parent) \
                                and (mz >= args.min_mz and mz <= args.max_mz) \
                                and (rt >= args.min_rt and rt <= args.max_rt) \
                                else False for name, mzp, mz, rt in zip(columns, mz_parents, mzs, rts)
                             ]
            matrix = matrix.loc[:, columns_to_keep]
            if args.features_selection != 'none':
                top_features = read_csv(csv_file=f"{path}/{args.features_file}", num_rows=-1, n_cols=args.n_features)
                matrix = matrix.loc[:, top_features.iloc[:, 0].values[:args.n_features]]
            else:
                matrix = matrix.iloc[:, :args.n_features]
            if args.remove_bad_samples:
                # if a file named bad_samples.csv in resources folder, remove those samples
                print("Removing bad samples")
                if os.path.exists(f"resources/bad_samples.csv"):
                    bad_samples = pd.read_csv(f"resources/bad_samples.csv", header=None).values.squeeze()
                    mask = np.array([x not in bad_samples for x in names])
                    notmask = np.array([x in bad_samples for x in names])
                    removed = np.array([x for x in names[notmask] if x in bad_samples])

                    # assert len(removed) == len(bad_samples) - 2  # 2 samples are in B15 which is not used

                    matrix = matrix.loc[mask]
                    names = names[mask]
                    labels = labels[mask]
                    batches = batches[mask]
                    orders = orders[mask]
                    concs = concs[mask]
                    manips = manips[mask]
                    urines = urines[mask]

                    print(f"Removed {removed}")
                    assert len(matrix) == len(names) == len(labels) == len(batches) == len(orders) == len(concs) == len(manips) == len(urines)
                    
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            if args.log1p:
                matrix.iloc[:] = np.log1p(matrix.values)
            matrix.iloc[:] = np.nan_to_num(matrix.values)

            pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' not in name]

            pos = [i for i, name in enumerate(names.values.flatten()) if 'urinespositives' in name]
            not_pos = [i for i, name in enumerate(names.values.flatten()) if 'urinespositives' not in name]
            data['inputs']["urinespositives"], data['inputs'][group] = matrix.iloc[pos], matrix.iloc[not_pos]
            data['names']["urinespositives"], data['names'][group] = names.to_numpy()[pos], names.to_numpy()[not_pos]
            data['labels']["urinespositives"], data['labels'][group] = labels.to_numpy()[pos], labels.to_numpy()[not_pos]
            data['batches']["urinespositives"], data['batches'][group] = batches.to_numpy()[pos], batches.to_numpy()[not_pos]
            data['manips']["urinespositives"], data['manips'][group] = manips.to_numpy()[pos], manips.to_numpy()[not_pos]
            data['urines']["urinespositives"], data['urines'][group] = urines.to_numpy()[pos], urines.to_numpy()[not_pos]
            data['concs']["urinespositives"], data['concs'][group] = concs.to_numpy()[pos], concs.to_numpy()[not_pos]
            data['orders']["urinespositives"], data['orders'][group] = orders[pos], orders[not_pos]

            unique_labels = np.array(np.unique(data['labels'][group]))
            # place blancs at the end
            blanc_class = np.argwhere(unique_labels == 'blanc').flatten()[0]
            unique_labels = np.concatenate((np.delete(unique_labels, blanc_class), ['blanc']))

            data['cats'][group] = np.array(
                [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])]
            )

            if args.pool:
                pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
                data['inputs'][f"{group}_pool"] = matrix.iloc[pool_pos]
                data['names'][f"{group}_pool"] = np.array([f'pool_{i}' for i, _ in enumerate(pool_pos)])
                data['labels'][f"{group}_pool"] = np.array([f'pool' for _ in pool_pos])
                data['batches'][f"{group}_pool"] = batches[pool_pos]
                data['manips'][f"{group}_pool"] = manips[pool_pos]
                data['urines'][f"{group}_pool"] = urines[pool_pos]
                data['concs'][f"{group}_pool"] = concs[pool_pos]

                # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
                data['orders'][f"{group}_pool"] = orders[pool_pos]
                data['cats'][f"{group}_pool"] = np.array(
                    [len(np.unique(data['labels'][group])) for _ in batches[pool_pos]])

                data['labels'][group] = np.array([x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = np.concatenate((unique_labels, np.array(['pool'])))
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])

    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
    if not args.pool:
        for key in list(data.keys()):
            if key in ['inputs']:
                data[key]['all'] = pd.concat((
                    data[key]['train'], data[key]['valid'], data[key]['test']
                ), axis=0)
            else:
                data[key]['all'] = np.concatenate((
                    data[key]['train'], data[key]['valid'], data[key]['test']
                ), axis=0)
        
        unique_batches = np.unique(data['batches']['all'])
        # for group in ['train', 'valid', 'test', 'all']:
        #     data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])
    else:
        for key in list(data.keys()):
            if key in ['inputs']:
                data[key]['all'] = pd.concat((
                    data[key]['train'], data[key]['valid'], data[key]['test'],
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), axis=0)
                data[key]['all_pool'] = pd.concat((
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), axis=0)
            else:
                data[key]['all'] = np.concatenate((
                    data[key]['train'], data[key]['valid'], data[key]['test'],
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)
                data[key]['all_pool'] = np.concatenate((
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)

        unique_batches = np.unique(data['batches']['all'])
        # for group in ['train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all', 'all_pool']:
        #     data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches, unique_manips, unique_urines, unique_concs


def get_data2(path, args, seed=42):
    """

    Args:
        path: Path where the csvs can be loaded. The folder designated by path needs to contain at least
                   one file named train_inputs.csv (when using --use_valid=0 and --use_test=0). When using
                   --use_valid=1 and --use_test=1, it must also contain valid_inputs.csv and test_inputs.csv.

    Returns:
        data
    """
    data = {}
    unique_labels = np.array([])
    for info in ['inputs', 'names', 'labels', 'cats', 'batches', 'manips', 'orders', 'sets', 'urines', 'concs']:
        data[info] = {}
        for group in ['all', 'train', 'test', 'valid']:
            data[info][group] = np.array([])
    for group in ['train', 'valid']:
        # print('GROUP:', group)
        if group == 'valid':
            if args.groupkfold:
                skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                # Remove samples from unwanted batches
                splitter = skf.split(train_nums, data['labels']['train'], data['batches']['train'])

            else:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                train_nums = np.arange(0, len(data['labels']['train']))
                splitter = skf.split(train_nums, data['labels']['train'])

            _, valid_inds = splitter.__next__()
            _, test_inds = splitter.__next__()
            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            data['inputs']['train'], data['inputs']['valid'], data['inputs']['test'] = data['inputs']['train'].iloc[train_inds], \
                data['inputs']['train'].iloc[valid_inds], data['inputs']['train'].iloc[test_inds]
            data['labels']['train'], data['labels']['valid'], data['labels']['test'] = data['labels']['train'][train_inds], \
                data['labels']['train'][valid_inds], data['labels']['train'][test_inds]
            data['names']['train'], data['names']['valid'], data['names']['test'] = data['names']['train'].iloc[train_inds], \
                data['names']['train'].iloc[valid_inds], data['names']['train'].iloc[test_inds]
            data['orders']['train'], data['orders']['valid'], data['orders']['test'] = data['orders']['train'][train_inds], \
                data['orders']['train'][valid_inds], data['orders']['train'][test_inds]
            data['batches']['train'], data['batches']['valid'], data['batches']['test'] = data['batches']['train'][train_inds], \
                data['batches']['train'][valid_inds], data['batches']['train'][test_inds]
            data['manips']['train'], data['manips']['valid'], data['manips']['test'] = data['manips']['train'][train_inds], \
                data['manips']['train'][valid_inds], data['manips']['train'][test_inds]
            data['cats']['train'], data['cats']['valid'], data['cats']['test'] = data['cats']['train'][train_inds], data['cats']['train'][
                valid_inds], data['cats']['train'][test_inds]
            data['urines']['train'], data['urines']['valid'], data['urines']['test'] = data['urines']['train'][train_inds], \
                data['urines']['train'][valid_inds], data['urines']['train'][test_inds]
            data['concs']['train'], data['concs']['valid'], data['concs']['test'] = data['concs']['train'][train_inds], data['concs']['train'][
                valid_inds], data['concs']['train'][test_inds]

            if args.pool:
                if args.groupkfold:
                    skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                    train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                    pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'],
                                                    data['batches']['train_pool'])

                else:
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                    train_nums_pool = np.arange(0, len(data['labels']['train_pool']))
                    pool_splitter = skf.split(train_nums_pool, data['labels']['train_pool'])

                _, valid_inds = pool_splitter.__next__()
                _, test_inds = pool_splitter.__next__()
                train_inds = [x for x in train_nums_pool if x not in np.concatenate((valid_inds, test_inds))]
                data['inputs']['train_pool'], data['inputs']['valid_pool'], data['inputs']['test_pool'], = data['inputs']['train_pool'].iloc[train_inds], \
                    data['inputs']['train_pool'].iloc[valid_inds], data['inputs']['train_pool'].iloc[test_inds]
                data['labels']['train_pool'], data['labels']['valid_pool'], data['labels']['test_pool'], = data['labels']['train_pool'][train_inds], \
                    data['labels']['train_pool'][valid_inds], data['labels']['train_pool'][test_inds]
                data['names']['train_pool'], data['names']['valid_pool'], data['names']['test_pool'], = data['names']['train_pool'][train_inds], \
                    data['names']['train_pool'][valid_inds], data['names']['train_pool'][test_inds]
                data['orders']['train_pool'], data['orders']['valid_pool'], data['orders']['test_pool'], = data['orders']['train_pool'][train_inds], \
                    data['orders']['train_pool'][valid_inds], data['orders']['train_pool'][test_inds]
                data['batches']['train_pool'], data['batches']['valid_pool'], data['batches']['test_pool'], = data['batches']['train_pool'][train_inds], \
                    data['batches']['train_pool'][valid_inds], data['batches']['train_pool'][test_inds]
                data['manips']['train_pool'], data['manips']['valid_pool'], data['manips']['test_pool'], = data['manips']['train_pool'][train_inds], \
                    data['manips']['train_pool'][valid_inds], data['manips']['train_pool'][test_inds]
                data['cats']['train_pool'], data['cats']['valid_pool'], data['cats']['test_pool'], = data['cats']['train_pool'][train_inds], data['cats']['train_pool'][
                    valid_inds], data['cats']['train_pool'][test_inds]
                data['urines']['train_pool'], data['urines']['valid_pool'], data['urines']['test_pool'], = data['urines']['train_pool'][train_inds], \
                    data['urines']['train_pool'][valid_inds], data['urines']['train_pool'][test_inds]
                data['concs']['train_pool'], data['concs']['valid_pool'], data['concs']['test_pool'], = data['concs']['train_pool'][train_inds], data['concs']['train_pool'][
                    valid_inds], data['concs']['train_pool'][test_inds]

        else:
            if args.low_ram:
                matrix = read_csv_low_ram(csv_file=f"{path}/{args.csv_file}", num_rows=-1, n_cols=args.n_features)
            else:
                matrix = read_csv(csv_file=f"{path}/{args.csv_file}", num_rows=-1, n_cols=args.n_features)
            top_features = read_csv(csv_file=f"{path}/{args.features_file}", num_rows=-1, n_cols=args.n_features)
            names = matrix.iloc[:, 0]
            labels = matrix.iloc[:, 1]
            print(len(names))
            # if a file named bad_samples.csv in resources folder, remove those samples
            if os.path.exists(f"resources/bad_samples.csv"):
                bad_samples = pd.read_csv(f"resources/bad_samples.csv", header=None).values.squeeze()
                mask = np.array([x not in bad_samples for x in names])
                matrix = matrix.loc[mask]
                names = names[mask]
                labels = labels[mask]
                batches = batches[mask]
                orders = orders[mask]

            batches = matrix.iloc[:, 2]
            manips = pd.Series([x.split("_")[2] for x in names])
            urines = pd.Series([x.split("_")[3] for x in names])
            concs = pd.Series([x.split("_")[4] for x in names])
            unique_batches = batches.unique()
            unique_manips = manips.unique()
            unique_urines = urines.unique()
            unique_concs = concs.unique()
            # batches = np.stack([np.argwhere(x == unique_batches).squeeze() for x in batches])
            orders = np.array([0 for _ in batches])
            matrix = matrix.iloc[:, 3:].fillna(0).astype(float)
            matrix = matrix.loc[:, top_features.iloc[:, 0].values[:args.n_features]]
            if args.remove_zeros:
                mask1 = (matrix == 0).mean(axis=0) < 0.1
                matrix = matrix.loc[:, mask1]
            if args.log1p:
                matrix.iloc[:] = np.log1p(matrix.values)
            # pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
            print(len(names))
            pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' not in name]
            print(len(pos))

            data['inputs'][group] = matrix.iloc[pos]
            data['names'][group] = names
            data['labels'][group] = labels.to_numpy()[pos]
            data['batches'][group] = batches[pos]
            data['manips'][group] = manips[pos]
            data['urines'][group] = urines[pos]
            data['concs'][group] = concs[pos]
            # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
            data['orders'][group] = orders[pos]
            
            unique_labels = np.array(np.unique(data['labels']['train']))
            # place blancs at the end
            blanc_class = np.argwhere(unique_labels == 'blanc').flatten()[0]
            unique_labels = np.concatenate((np.delete(unique_labels, blanc_class), ['blanc']))
            data['cats'][group] = np.array(
                [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])

            if args.pool:
                pool_pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' in name]
                data['inputs'][f"{group}_pool"] = matrix.iloc[pool_pos]
                data['names'][f"{group}_pool"] = np.array([f'pool_{i}' for i, _ in enumerate(pool_pos)])
                data['labels'][f"{group}_pool"] = np.array([f'pool' for _ in pool_pos])
                data['batches'][f"{group}_pool"] = batches[pool_pos]
                data['manips'][f"{group}_pool"] = manips[pool_pos]
                data['urines'][f"{group}_pool"] = urines[pool_pos]
                data['concs'][f"{group}_pool"] = concs[pool_pos]

                # This is juste to make the pipeline work. Meta should be 0 for the amide dataset
                data['orders'][f"{group}_pool"] = orders[pool_pos]
                data['cats'][f"{group}_pool"] = np.array(
                    [len(np.unique(data['labels'][group])) for _ in batches[pool_pos]])

                data['labels'][group] = np.array([x.split('-')[0] for i, x in enumerate(data['labels'][group])])
                unique_labels = np.concatenate((get_unique_labels(data['labels'][group]), np.array(['pool'])))
                data['cats'][group] = np.array(
                    [np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels'][group])])

    for key in list(data['names'].keys()):
        data['sets'][key] = np.array([key for _ in data['names'][key]])
        # print(key, data['sets'][key])
    if not args.pool:
        for key in list(data.keys()):
            if key in ['inputs']:
                data[key]['all'] = pd.concat((
                    data[key]['train'], data[key]['valid'], data[key]['test']
                ), 0)
            else:
                data[key]['all'] = np.concatenate((
                    data[key]['train'], data[key]['valid'], data[key]['test']
                ), 0)
        
        unique_batches = np.unique(data['batches']['all'])
        # for group in ['train', 'valid', 'test', 'all']:
        #     data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])
    else:
        # print('POOL!!')
        for key in list(data.keys()):
            # print('key', key)
            if key in ['inputs']:
                data[key]['all'] = pd.concat((
                    data[key]['train'], data[key]['valid'], data[key]['test'],
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)
                data[key]['all_pool'] = pd.concat((
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)
            else:
                data[key]['all'] = np.concatenate((
                    data[key]['train'], data[key]['valid'], data[key]['test'],
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)
                data[key]['all_pool'] = np.concatenate((
                    data[key]['train_pool'], data[key]['valid_pool'], data[key]['test_pool'],
                ), 0)

        unique_batches = np.unique(data['batches']['all'])
        # for group in ['train', 'valid', 'test', 'train_pool', 'valid_pool', 'test_pool', 'all', 'all_pool']:
        #     data['batches'][group] = np.array([np.argwhere(unique_batches == x)[0][0] for x in data['batches'][group]])

    return data, unique_labels, unique_batches, unique_manips, unique_urines, unique_concs

