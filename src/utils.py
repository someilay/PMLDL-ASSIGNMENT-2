import numpy as np
import pandas as pd

import torch

from typing import Union, Iterable
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder


def get_not_in(x: Iterable[int], low: int, high: int) -> int:
    candidate = np.random.randint(low, high)
    while candidate in x:
        candidate = np.random.randint(low, high)
    return candidate


def get_batch(data: pd.DataFrame,
              batch_size: int,
              n_usr: int,
              n_itm: int,
              dev: Union[torch.device, str] = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return merged_data.user_id_new.nunique(), merged_data.item_id_new.nunique()


def load_merged_data(merged_path: Path) -> pd.DataFrame:
    return pd.read_csv(merged_path, sep="\t")


def get_user_features(data: pd.DataFrame, dev: Union[torch.device, str] = "cpu") -> torch.Tensor:
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
    item_features_df = data[f_cols + ['item_id_new']].sort_values('item_id_new')
    item_features_df = item_features_df.groupby('item_id_new').first().reset_index()
    item_features_df = item_features_df[f_cols]
    item_features = torch.tensor(item_features_df.values, dtype=torch.float)

    return item_features.to(dev)
