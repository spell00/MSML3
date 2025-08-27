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
    """Fast CSV reader specialized for current dataset layout.

    Assumptions:
      - First 3 columns are meta (kept as object/string)
      - Remaining columns are numeric features
      - Parameter n_cols refers to number of *feature* columns (excluding the 3 meta cols).
        Use -1 to load all feature columns.
      - If num_rows == -1 load all rows.

    Improvements over previous implementation:
      - Single fast line count (no re-opening inside loop)
      - Two-pass (count -> parse) instead of repeated counting per row
      - Avoid per-row numpy array creation; direct float conversion
      - Proper tqdm usage with known total (reduces overhead & gives ETA)
      - Keeps numeric block as float array; only meta columns are object
    """
    # -------- Helper: fast line count (bytes based) --------
    def _count_lines(path):
        cnt = 0
        with open(path, 'rb') as fb:
            for blk in iter(lambda: fb.read(1024 * 1024), b''):
                cnt += blk.count(b'\n')
        return cnt

    # ---------- Discover header & dimensions ----------
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header_full = next(reader)

    total_lines = _count_lines(csv_file)  # includes header line if file ends with \n (common). We'll adjust.
    # If file does not end with newline, last line still counted by .count("\n") logic -> subtract header explicitly
    data_lines = max(total_lines - 1, 0)
    if num_rows == -1:
        target_rows = data_lines
    else:
        target_rows = min(num_rows, data_lines)

    # Determine feature column subset
    if n_cols == -1:
        header = header_full  # all columns
        feature_cols = header_full[3:]
    else:
        # Keep first 3 + next n_cols feature columns (truncate if fewer available)
        feature_limit = min(n_cols, max(len(header_full) - 3, 0))
        header = header_full[:3 + feature_limit]
        feature_cols = header[3:]

    n_features = len(feature_cols)
    dtype_num = np.float16 if fp == 'float16' else np.float32

    # Preallocate
    meta_arr = np.empty((target_rows, 3), dtype=object)
    num_arr = np.empty((target_rows, n_features), dtype=dtype_num)

    # ---------- Second pass: parse ----------
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        pbar = tqdm(total=target_rows, desc="Reading CSV", disable=(target_rows < 1000))
        for i, row in enumerate(reader):
            if i >= target_rows:
                break
            # Meta columns
            meta_arr[i, 0] = row[0]
            meta_arr[i, 1] = row[1]
            meta_arr[i, 2] = row[2]
            # Numeric features slice boundaries
            if n_features:
                # Row may be shorter than expected if malformed; guard length
                # Start index 3, end index 3 + n_features
                raw_vals = row[3: 3 + n_features]
                # Fill numeric array; convert '' to 0.0
                # Manual loop usually faster than creating intermediate numpy array for large width
                for j, v in enumerate(raw_vals):
                    if v == '' or v is None:
                        num_arr[i, j] = 0.0
                    else:
                        try:
                            num_arr[i, j] = float(v)
                        except ValueError:
                            # Fallback if non-numeric token: treat as 0.0
                            num_arr[i, j] = 0.0
                # If provided fewer columns than expected (rare), pad remaining with 0
                if len(raw_vals) < n_features:
                    num_arr[i, len(raw_vals):] = 0.0
            pbar.update(1)
        pbar.close()

    meta_df = pd.DataFrame(meta_arr, columns=header[:3])
    if n_features:
        num_df = pd.DataFrame(num_arr, columns=feature_cols)
        df = pd.concat([meta_df, num_df], axis=1)
    else:
        df = meta_df
    return df


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
        labels = pd.Series(['blanc' if label == 'blanc' else 'bact' for label in labels])

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
    columns_to_keep = [True if (mzp >= args.min_mz_parent and mzp <= args.max_mz_parent)
                       and (mz >= args.min_mz and mz <= args.max_mz)
                       and (rt >= args.min_rt and rt <= args.max_rt)
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
        data['orders']["test_pool"] = orders[pool_pos]
        data['cats']["test_pool"] = np.array(
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
        # Optional: load only XGBoost-selected features directly from CSV
        xgb_top_features = None
        if getattr(args, 'xgboost_features', 1):
            # Attempt to locate an xgboost_feature_importance_*.csv file heuristically
            # Pattern (simplified vs select_xgboost_features): search under results/multi/... for matching cropping & n_features
            try:
                import re
                importance_candidates = []
                base_results_dir = os.path.join(
                    'results', 'multi', f"mz{args.mz}", f"rt{args.rt}", f"ms{args.ms_level}", f"{args.spd}spd"
                )
                if os.path.isdir(base_results_dir):
                    for root, dirs, files in os.walk(base_results_dir):
                        if 'xgboost_feature_importance_' in '\n'.join(files) and 'xgboost' in root:
                            for fn in files:
                                if fn.startswith('xgboost_feature_importance_zscore') and fn.endswith('.csv'):
                                    fullp = os.path.join(root, fn)
                                    # Heuristic filters: threshold 0.0 or current threshold, contains n_features & cropping
                                    if f"mz{args.min_mz}-{args.max_mz}rt{args.min_rt}-{args.max_rt}" in root and \
                                       f"binary{args.binary}" in root and \
                                       f"gkf{args.groupkfold}" in root and \
                                       f"_{args.n_features}_" if args.n_features > 0 else True:
                                        importance_candidates.append(fullp)
                if not importance_candidates:
                    # Fallback: broader search without strict filters
                    if os.path.isdir(base_results_dir):
                        for root, dirs, files in os.walk(base_results_dir):
                            for fn in files:
                                if fn.startswith('xgboost_feature_importance_') and fn.endswith('.csv'):
                                    importance_candidates.append(os.path.join(root, fn))
                if importance_candidates:
                    # Pick most recent
                    importance_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    feat_imp_path = importance_candidates[0]
                    feat_imp_df = pd.read_csv(feat_imp_path)
                    if 'feature' in feat_imp_df.columns:
                        # Order by importance descending if column exists
                        if args.n_features > 0:
                            xgb_top_features = feat_imp_df.nlargest(args.n_features, 'importance')['feature'].values
                        else:
                            xgb_top_features = feat_imp_df.nlargest(feat_imp_df.shape[0], 'importance')['feature'].values
                        if args.min_features_importance > 0:
                            xgb_large_features = feat_imp_df[feat_imp_df['importance'] > args.min_features_importance].loc[:, 'feature'].values
                        else:
                            xgb_large_features = feat_imp_df.loc[:, 'feature'].values
                        xgb_top_features = np.intersect1d(xgb_top_features, xgb_large_features)
                        print(f"[get_data_all] Loaded {len(xgb_top_features)} XGBoost features from {feat_imp_path}")
                else:
                    print("[get_data_all] No XGBoost feature importance file found; falling back to full CSV load.")
            except Exception as e:
                print(f"[get_data_all] Error while searching XGBoost features: {e}. Fallback to full CSV.")
                xgb_top_features = None

        if xgb_top_features is not None:
            matrix = read_csv_selected(
                csv_file=f"{path}/{args.csv_file}",
                selected_features=xgb_top_features,
                num_rows=-1,
                fp=args.fp,
                base_cols=3,
                fill_missing=0.0,
            )
        else:
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
        columns_to_keep = [True if (mzp >= args.min_mz_parent and mzp <= args.max_mz_parent)
                           and (mz >= args.min_mz and mz <= args.max_mz)
                           and (rt >= args.min_rt and rt <= args.max_rt)
                           else False for name, mzp, mz, rt in zip(columns, mz_parents, mzs, rts)
                           ]
        matrix = matrix.loc[:, columns_to_keep]
        if xgb_top_features is not None:
            # Already projected; nothing else to do. If fewer than requested, keep as is.
            pass
        elif args.features_selection != 'none':
            top_features = read_csv(csv_file=f"{path}/{args.features_file}", num_rows=-1, n_cols=args.n_features, fp=args.fp)
            matrix = matrix.loc[:, top_features.iloc[:, 0].values[:args.n_features]]
        else:
            if args.n_features != -1:
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
        unique_labels = np.concatenate((['blanc'], np.delete(unique_labels, blanc_class)))

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
            pd.read_csv('resources/bacteries_2024/BPatients-14-03-2025/patients_samples_20250318.csv'),
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


def read_csv_selected(csv_file, selected_features, num_rows=-1, fp='float32', base_cols=3, fill_missing=0.0):
    """
    Efficiently read only selected feature columns (plus first base_cols meta columns).
    selected_features: ordered list of feature column names you want (e.g. from XGBoost).
    """

    # First pass: fast line count with progress + read header
    # We count newlines in binary mode to avoid decoding overhead, with a tqdm over bytes.
    file_size = os.path.getsize(csv_file)
    newline_count = 0
    with open(csv_file, 'rb') as fb:
        # Disable progress bar for small files (<5MB) to reduce overhead
        pbar = tqdm(total=file_size, desc="Counting lines", unit='B', unit_scale=True, disable=(file_size < 5_000_000))
        block_size = 8 * 1024 * 1024  # 8MB blocks
        last_byte_newline = True
        while True:
            chunk = fb.read(block_size)
            if not chunk:
                break
            pbar.update(len(chunk))
            newline_count += chunk.count(b'\n')
            last_byte_newline = chunk.endswith(b'\n')
        pbar.close()
    # If file doesn't end with newline and is non-empty, add one line
    total_lines = newline_count if last_byte_newline or newline_count == 0 else newline_count + 1
    # Read header separately (text mode) and adjust sample count
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    if num_rows == -1:
        num_rows = total_lines  # includes header
    else:
        num_rows = min(num_rows, total_lines)
    n_samples = max(num_rows - 1, 0)

    meta_cols = header[:base_cols]
    header_features = header[base_cols:]

    # Map feature name -> index (relative to full row)
    feat_pos = {name: base_cols + i for i, name in enumerate(header_features)}

    # Build ordered target feature list (truncate to provided length)
    target_feats = list(selected_features)
    n_feats = len(target_feats)

    # Positions array: -1 means missing → fill with zeros
    positions = np.array([feat_pos.get(name, -1) for name in target_feats], dtype=np.int32)

    # Allocate arrays
    meta_str = np.empty((n_samples, base_cols), dtype=object)
    dtype_num = np.float16 if fp == 'float16' else np.float32
    data_num = np.empty((n_samples, n_feats), dtype=dtype_num)

    # Second pass: fill
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        pbar = tqdm(total=n_samples, desc="Reading selected CSV", disable=(n_samples < 1000))
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            # meta
            for m in range(base_cols):
                meta_str[i, m] = row[m]
            # numeric selected
            r_local = row  # local ref
            for j, pos in enumerate(positions):
                if pos == -1:
                    data_num[i, j] = fill_missing
                else:
                    val = r_local[pos]
                    data_num[i, j] = 0.0 if val == '' else float(val)
            pbar.update(1)
        pbar.close()
    # Combine
    df = pd.DataFrame(
        np.concatenate([meta_str, data_num], axis=1),
        columns=meta_cols + target_feats
    )
    return df
