import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
import os


def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b:
            break
        yield b


def read_csv(csv_file, num_rows=1000, n_cols=1000, fp="float64"):
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
            if n_cols != -1:
                row = np.array(row)[:n_cols]
            else:
                row = np.array(row)
            if row_num == 0:
                if num_rows == -1:
                    num_rows = sum(1 for _ in open(csv_file, 'rb'))
                else:
                    num_rows = min(num_rows, sum(1 for _ in open(csv_file, 'rb')))
                if n_cols != -1:
                    row = np.array(row)[:n_cols]
                data_num = np.empty((num_rows-1, len(row)-3))
                data_str = np.empty((num_rows-1, 3), dtype=object)
                header = row
                continue
            if num_rows != -1:
                if row_num >= num_rows:
                    break

            # Replace '' by 0 in row
            row = np.array([0 if x == '' else x for x in row])
            if fp == 'float16':
                data_num[row_num-1] = row[3:].astype(np.float16)
            else:
                data_num[row_num-1] = row[3:].astype(np.float32)
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
    for info in ['inputs', 'names', 'labels', 'cats', 'batches', 'batches_labels', 'manips', 'orders', 'sets', 'urines', 'concs']:
        data[info] = {}
    data[info]['all'] = data[info]['test'] = data[info]['urinespositives'] = np.array([])
    matrix = read_csv(csv_file=f"{path}/{args.csv_file}",
                      num_rows=-1,
                      n_cols=args.n_features,
                      fp=args.fp)
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
    data['batches_labels']["urinespositives"], data['batches_labels']['test'] =\
        batches.to_numpy()[pos], batches.to_numpy()[not_pos]
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
        data['inputs']["test_pool"] = matrix.iloc[pool_pos]
        data['names']["test_pool"] = np.array(['pool_{i}' for i, _ in enumerate(pool_pos)])
        data['labels']["test_pool"] = np.array(['pool' for _ in pool_pos])
        data['batches']["test_pool"] = batches[pool_pos]
        data['batches_labels']["test_pool"] = batches.to_numpy()[pos]
        data['manips']["test_pool"] = manips[pool_pos]
        data['urines']["test_pool"] = urines[pool_pos]
        data['concs']["test_pool"] = concs[pool_pos]

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


def get_data_all(path, args, seed=42):
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
    for info in ['inputs', 'names', 'labels', 'cats', 'batches', 'batches_labels', 'manips', 'orders',
                 'urines', 'concs']:
        data[info] = {}
        for group in ['all']:
            data[info][group] = np.array([])
    for group in ['all']:
        if args.low_ram:
            matrix = read_csv_low_ram(csv_file=f"{path}/{args.csv_file}", num_rows=-1, n_cols=args.n_features)
        else:
            matrix = read_csv(csv_file=f"{path}/{args.csv_file}", num_rows=-1, n_cols=args.n_features, fp=args.fp)
        matrix.index = names = matrix.iloc[:, 0]
        names = matrix.iloc[:, 0]
        labels = matrix.iloc[:, 1]
        batches = matrix.iloc[:, 2]
        manips = pd.Series([x.split("_")[2] for x in names])
        urines = pd.Series([x.split("_")[3] for x in names])
        concs = pd.Series([x.split("_")[4] if len(x.split("_")) == 5 else 'na' for x in names])
        unique_batches = batches.unique()
        unique_manips = manips.unique()
        unique_urines = urines.unique()
        unique_concs = concs.unique()
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
            top_features = read_csv(csv_file=f"{path}/{args.features_file}", num_rows=-1, n_cols=args.n_features, fp=args.fp)
            matrix = matrix.loc[:, top_features.iloc[:, 0].values[:args.n_features]]
        else:
            matrix = matrix.iloc[:, :args.n_features]
        if args.remove_bad_samples:
            # if a file named bad_samples.csv in resources folder, remove those samples
            print("Removing bad samples")
            if os.path.exists("resources/bad_samples.csv"):
                bad_samples = pd.read_csv("resources/bad_samples.csv", header=None).values.squeeze()
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
        batches_labels = batches.copy()
        batches.iloc[:] = pd.Series([np.argwhere(x == unique_batches).squeeze() for x in batches])

        pos = [i for i, name in enumerate(names.values.flatten()) if 'QC' not in name]

        pos = [i for i, name in enumerate(names.values.flatten()) if 'urinespositives' in name]
        not_pos = [i for i, name in enumerate(names.values.flatten()) if 'urinespositives' not in name]
        data['inputs']["urinespositives"], data['inputs'][group] = matrix.iloc[pos], matrix.iloc[not_pos]
        data['names']["urinespositives"], data['names'][group] = names.to_numpy()[pos], names.to_numpy()[not_pos]
        data['labels']["urinespositives"], data['labels'][group] = labels.to_numpy()[pos], labels.to_numpy()[not_pos]
        data['batches_labels']["urinespositives"], data['batches_labels'][group] =\
            batches_labels.to_numpy()[pos], batches_labels.to_numpy()[not_pos]
        data['batches']["urinespositives"], data['batches'][group] =\
            np.stack(batches.to_numpy()[pos]), np.stack(batches.to_numpy()[not_pos])
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
            data['labels'][f"{group}_pool"] = np.array(['pool' for _ in pool_pos])
            data['batches'][f"{group}_pool"] = batches[pool_pos]
            data['batches_labels'][f"{group}_pool"] = batches[pool_pos]
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

        urinespositives_names = np.array([x.split('_')[-2] for x in data['names']['urinespositives']])
        # TODO load expected classes elsewhere
        urinespositives_real_df = pd.concat((
            pd.read_csv('resources/bacteries_2024/B10-05-03-2024/b10_patients_samples.csv'),
            pd.read_csv('resources/bacteries_2024/B11-05-24-2024/b11_patients_samples.csv'),
            pd.read_csv('resources/bacteries_2024/BPatients-03-14-2025/patients_samples_20250318.csv'),
        ))
        # Keep unique samples
        urinespositives_real_df = urinespositives_real_df.drop_duplicates(subset=['ID'])
        urinespositives_real_df.loc[:, 'Class'] = [
            label.lower() for label in urinespositives_real_df.loc[:, 'Class'].to_numpy()
        ]
        urinespositives_inds = np.array([
            i for i, x in enumerate(urinespositives_names) if x in urinespositives_real_df.loc[:, 'ID'].to_numpy()
        ])
        # Get the labels of urinespositives_real_df in the same order as the names
        for k in data.keys():
            if k == 'inputs':
                data[k]['urinespositives'] = data[k]['urinespositives'].iloc[urinespositives_inds, :]
            elif k in ['cats']:
                data[k]['urinespositives'] = []
            else:
                data[k]['urinespositives'] = data[k]['urinespositives'][urinespositives_inds]

        data['labels']['urinespositives'] = np.array([
            urinespositives_real_df.loc[
                urinespositives_real_df.loc[:, 'ID'] == x, 'Class'
            ].to_numpy()[0] for x in urinespositives_names[urinespositives_inds]
        ])
        # find the labels that are not in the unique labels
        unique_labels_set = set(unique_labels)
        labels_to_remove = [label for label in data['labels']['urinespositives'] if label not in unique_labels_set]
        inds_to_remove = [i for i, x in enumerate(data['labels']['urinespositives']) if x in labels_to_remove]
        inds_to_keep = [i for i, x in enumerate(data['labels']['urinespositives']) if x not in labels_to_remove]
        # Urines are present in the batch bpatients. Remove them in the bpatients

        seen = set()
        urines_to_keep = []
        for i in reversed(range(len(data['urines']['urinespositives']))):
            x = data['urines']['urinespositives'][i]
            if x not in seen:
                seen.add(x)
                urines_to_keep.insert(0, x)  # insert at front to maintain reverse order
                inds_to_keep.insert(0, i)    # same for indices
            else:
                inds_to_remove.insert(0, i)

        inds_to_remove = np.unique(inds_to_remove)
        inds_to_keep = np.array([i for i in range(len(data['labels']['urinespositives'])) if i not in inds_to_remove])
        # Remove the labels that are not in the unique labels
        data['batches_labels']['urinespositives'] = np.delete(data['batches_labels']['urinespositives'], inds_to_remove)
        data['labels']['urinespositives'] = np.delete(data['labels']['urinespositives'], inds_to_remove)
        data['batches']['urinespositives'] = np.delete(data['batches']['urinespositives'], inds_to_remove)
        data['manips']['urinespositives'] = np.delete(data['manips']['urinespositives'], inds_to_remove)
        data['urines']['urinespositives'] = np.delete(data['urines']['urinespositives'], inds_to_remove)
        data['concs']['urinespositives'] = np.delete(data['concs']['urinespositives'], inds_to_remove)
        data['orders']['urinespositives'] = np.delete(data['orders']['urinespositives'], inds_to_remove)
        data['names']['urinespositives'] = np.delete(data['names']['urinespositives'], inds_to_remove)
        # Remove the samples from the inputs
        data['inputs']['urinespositives'] = data['inputs']['urinespositives'].iloc[inds_to_keep, :]

        data['cats']['urinespositives'] = np.array([
            np.where(x == unique_labels)[0][0] for i, x in enumerate(data['labels']['urinespositives'])
        ])
        try:
            assert sum([name.split('_')[-2] for name in data['names']['urinespositives']] == data['urines']['urinespositives']) == len(data['urines']['urinespositives'])
        except AssertionError as e:
            print(e)
            print(data['names']['urinespositives'])
            print(data['urines']['urinespositives'])
            raise ValueError("The urinespositives names and urines are not the same")

        if args.binary:
            urinespositives_real_df.loc[:, 'Class'] = [
                'blanc' if label == 'blanc' else 'bact' for label in urinespositives_real_df.loc[:, 'Class'].to_numpy()
            ]

    uniques = {
        'labels': unique_labels,
        'batches': unique_batches,
        'manips': unique_manips,
        'urines': unique_urines,
        'concs': unique_concs,
    }

    return data, uniques
