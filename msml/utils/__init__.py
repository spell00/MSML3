import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.pipeline import Pipeline


def scale_data(scale, data, device='cpu'):
    unique_batches = np.unique(data['batches']['all'])
    scaler = None
    if scale == 'binarize':
        for group in list(data['inputs'].keys()):
            data['inputs'][group] = data['inputs'][group]
            data['inputs'][group][data['inputs'][group] > 0] = 1
            data['inputs'][group][data['inputs'][group] <= 0] = 0

    elif scale == 'robust_per_batch':
        scalers = {b: RobustScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = RobustScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'standard_per_batch':
        scalers = {b: StandardScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = StandardScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'minmax_per_batch':
        scalers = {b: MinMaxScaler() for b in unique_batches}
        for b in unique_batches:
            mask = data['batches']['all'] == b
            scalers[b].fit(data['inputs']['all'][mask])
            for group in list(data['inputs'].keys()):
                if data['inputs'][group].shape[0] > 0:
                    mask = data['batches'][group] == b
                    # columns = data['inputs'][group][mask].columns
                    # indices = data['inputs'][group][mask].index
                    if data['inputs'][group][mask].shape[0] > 0:
                        data['inputs'][group].iloc[mask] = scalers[b].transform(data['inputs'][group][mask].to_numpy())
                    else:
                        data['inputs'][group][mask] = data['inputs'][group][mask]
            # data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)
        scaler = MinMaxScaler()
        scaler.fit(data['meta']['all'])
        for group in list(data['meta'].keys()):
            if data['meta'][group].shape[0] > 0:
                columns = data['meta'][group].columns
                indices = data['meta'][group].index
                if data['meta'][group].shape[0] > 0:
                    data['meta'][group] = scaler.transform(data['meta'][group].to_numpy())
                else:
                    data['meta'][group] = data['meta'][group]
                data['meta'][group] = pd.DataFrame(data['meta'][group], columns=columns, index=indices)

    elif scale == 'robust':
        scaler = RobustScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'robust_minmax':
        scaler = Pipeline([('robust', RobustScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard':
        scaler = StandardScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'standard_minmax':
        scaler = Pipeline([('standard', StandardScaler()), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'minmax':
        scaler = MinMaxScaler()
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l1_minmax':
        scaler = Pipeline([('l1', Normalizer(norm='l1')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2_minmax':
        scaler = Pipeline([('l2', Normalizer(norm='l2')), ('minmax', MinMaxScaler())])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l1':
        scaler = Pipeline([('l1', Normalizer(norm='l1'))])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'l2':
        scaler = Pipeline([('l2', Normalizer(norm='l2'))])
        for data_type in ['inputs', 'meta']:
            scaler.fit(data[data_type]['all'])
            for group in list(data[data_type].keys()):
                if data[data_type][group].shape[0] > 0:
                    columns = data[data_type][group].columns
                    indices = data[data_type][group].index
                    if data[data_type][group].shape[0] > 0:
                        data[data_type][group] = scaler.transform(data[data_type][group].to_numpy())
                    else:
                        data[data_type][group] = data[data_type][group]
                    data[data_type][group] = pd.DataFrame(data[data_type][group], columns=columns, index=indices)

    elif scale == 'none':
        return data, 'none'
    # Values put on [0, 1] interval to facilitate autoencoder reconstruction (enable use of sigmoid as final activation)
    # scaler = MinMaxScaler()
    # scaler.fit(data['inputs']['all'])
    # for group in list(data['inputs'].keys()):
    #     if data['inputs'][group].shape[0] > 0:
    #         columns = data['inputs'][group].columns
    #         indices = data['inputs'][group].index
    #         if data['inputs'][group].shape[0] > 0:
    #             data['inputs'][group] = scaler.transform(data['inputs'][group].to_numpy())
    #         else:
    #             data['inputs'][group] = data['inputs'][group]
    #         data['inputs'][group] = pd.DataFrame(data['inputs'][group], columns=columns, index=indices)

    return data, scaler
