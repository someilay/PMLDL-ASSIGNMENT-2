import numpy as np
import pandas as pd

import torch

from typing import Union, Iterable
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder


def get_not_in(x: Iterable[int], low: int, high: int) -> int:
    """
    Get a random integer not present in the input iterable.

    Parameters:
        x (Iterable[int]): Input iterable.
        low (int): Lower bound for random integer.
        high (int): Upper bound for random integer.

    Returns:
        int: Random integer not present in the input iterable.
    """
    candidate = np.random.randint(low, high)
    while candidate in x:
        candidate = np.random.randint(low, high)
    return candidate


def get_batch(data: pd.DataFrame,
              batch_size: int,
              n_usr: int,
              n_itm: int,
              dev: Union[torch.device, str] = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get a batch of training data for the GNN model.

    Parameters:
        data (pd.DataFrame): Merged DataFrame containing user-item interactions.
        batch_size (int): Size of the training batch.
        n_usr (int): Number of unique users.
        n_itm (int): Number of unique items.
        dev (Union[torch.device, str]): Device for training (cpu or cuda).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing tensors for chosen users,
            positive items, and negative items.
    """
    users_liked_items = data.groupby('user_id_new')['item_id_new'].apply(list).reset_index()
    indices: np.ndarray = np.arange(0, n_usr, 1)

    chosen_users = np.random.choice(indices, batch_size, replace=n_usr < batch_size)
    chosen_users.sort()

    users_liked_items = pd.merge(
        users_liked_items,
        pd.DataFrame(chosen_users, columns=['user_id_new']),
        how='right',
        on='user_id_new'
    )
    pos_items = users_liked_items['item_id_new'].apply(lambda x: np.random.choice(x)).values + n_usr
    neg_items = users_liked_items['item_id_new'].apply(lambda x: get_not_in(x, 0, n_itm)).values + n_usr

    return (
        torch.LongTensor(chosen_users).to(dev),
        torch.LongTensor(pos_items).to(dev),
        torch.LongTensor(neg_items).to(dev),
    )


def get_edge_index(data: pd.DataFrame,
                   n_usr: int,
                   dev: Union[torch.device, str] = 'cpu') -> torch.Tensor:
    """
    Get edge indices for the GNN model.

    Parameters:
        data (pd.DataFrame): Merged DataFrame containing user-item interactions.
        n_usr (int): Number of unique users.
        dev (Union[torch.device, str]): Device for training (cpu or cuda).

    Returns:
        torch.Tensor: Edge indices' tensor.
    """
    u_t = torch.LongTensor(data.user_id_new.values)
    i_t = torch.LongTensor(data.item_id_new.values) + n_usr

    train_edge_index = torch.stack(
        (
            torch.cat([u_t, i_t]),
            torch.cat([i_t, u_t])
        )
    ).to(dev)
    return train_edge_index


def count_user_items(merged_data: pd.DataFrame) -> tuple[int, int]:
    """
    Count the number of unique users and items in the merged data.

    Parameters:
        merged_data (pd.DataFrame): Merged DataFrame containing user-item interactions.

    Returns:
        tuple[int, int]: Number of unique users and items.
    """
    return merged_data.user_id_new.nunique(), merged_data.item_id_new.nunique()


def load_merged_data(merged_path: Path) -> pd.DataFrame:
    """
    Load merged data from a file.

    Parameters:
        merged_path (Path): Path to the merged data file.

    Returns:
        pd.DataFrame: Merged DataFrame containing user-item interactions.
    """
    return pd.read_csv(merged_path, sep="\t")


def get_user_features(data: pd.DataFrame, dev: Union[torch.device, str] = "cpu") -> torch.Tensor:
    """
    Get user features from the merged data.

    Parameters:
        data (pd.DataFrame): Merged DataFrame containing user-item interactions.
        dev (Union[torch.device, str]): Device for training (cpu or cuda).

    Returns:
        torch.Tensor: User features tensor.
    """
    encoder_1 = OneHotEncoder()
    user_occupation = data.groupby(['user_id_new', 'occupation']).first().reset_index()
    user_occupation = user_occupation[['user_id_new', 'occupation']]
    user_occupation = user_occupation.sort_values('user_id_new')

    user_features = np.expand_dims(user_occupation['occupation'].values, axis=1)
    user_features = encoder_1.fit_transform(user_features).toarray()
    user_features = torch.tensor(user_features, dtype=torch.float)

    return user_features.to(dev)


def get_item_features(data: pd.DataFrame,
                      f_cols: list[str],
                      dev: Union[torch.device, str] = "cpu") -> torch.Tensor:
    """
    Get item features from the merged data.

    Parameters:
        data (pd.DataFrame): Merged DataFrame containing user-item interactions.
        f_cols (list[str]): List of columns representing item features.
        dev (Union[torch.device, str]): Device for training (cpu or cuda).

    Returns:
        torch.Tensor: Item features' tensor.
    """
    item_features_df = data[f_cols + ['item_id_new']].sort_values('item_id_new')
    item_features_df = item_features_df.groupby('item_id_new').first().reset_index()
    item_features_df = item_features_df[f_cols]
    item_features = torch.tensor(item_features_df.values, dtype=torch.float)

    return item_features.to(dev)
