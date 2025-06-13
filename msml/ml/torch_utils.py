import numpy as np
import pandas as pd
import torch
import copy
import torch.nn.functional as F


def augment_data(X_train, n_aug, p=0, g=0):
    torch.manual_seed(42)
    X_train2 = copy.deepcopy(X_train)
    # y_train2 = np.array([])
    # X_train2 = np.array([])

    if n_aug > 0:
        for _ in range(n_aug):
            tmp = copy.deepcopy(X_train) + g * np.random.normal(0, 1, X_train.shape).astype(np.float32)
            tmp = F.dropout(torch.Tensor(tmp.to_numpy()), p).detach().cpu().numpy()
            if len(X_train2) > 0:
                X_train2 = np.concatenate([X_train2, tmp], 0)
            else:
                X_train2 = tmp
        columns = X_train.columns
        train_indices = np.array(X_train.index)
        # Check for duplicated indices
        if np.sum(X_train.index.duplicated()) > 0:
            duplicated_indices = np.argwhere(X_train.index.duplicated())[0]
            for ind in duplicated_indices:
                train_indices[ind] = f"{train_indices[ind]}_0"

        # train_indices = np.concatenate([train_indices] * (n_aug + 1))
        # Add copy num, 0 for original
        train_indices = [f"{x}_{i}" for i in range(n_aug + 1) for x in train_indices]

        assert len(train_indices) == np.unique(train_indices).shape[0]

        X_train2 = pd.DataFrame(X_train2, columns=columns, index=train_indices)

    return X_train2


def augment_data2(X_train, n_aug, p=0, g=0):
    torch.manual_seed(42)
    X_train2 = np.array([])

    if n_aug > 0:
        for _ in range(n_aug):
            tmp = copy.deepcopy(X_train) + g * np.random.normal(0, 1, X_train.shape).astype(np.float32)
            tmp = F.dropout(torch.Tensor(tmp.to_numpy()), p).detach().cpu().numpy()
            if len(X_train2) > 0:
                X_train2 = np.concatenate([X_train2, tmp], 0)
            else:
                X_train2 = tmp
    return X_train2


def to_categorical(y, num_classes):
    """
    1-hot encodes a tensor
    Args:
        y: values to encode
        num_classes: Number of classes. Length of the 1-encoder

    Returns:
        Tensor corresponding to the one-hot encoded classes
    """
    return torch.eye(num_classes, dtype=torch.int)[y]
