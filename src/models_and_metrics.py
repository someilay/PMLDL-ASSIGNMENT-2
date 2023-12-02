import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as torch_nnf

from typing import Union, Iterable
from pathlib import Path
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from tqdm import tqdm


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


def compute_bpr_loss(users: torch.Tensor,
                     users_emb: torch.Tensor,
                     pos_emb: torch.Tensor,
                     neg_emb: torch.Tensor,
                     user_emb_0: torch.Tensor,
                     pos_emb_0: torch.Tensor,
                     neg_emb_0: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
    # compute loss from initial embeddings, used for regularization
    reg_loss = (1 / 2) * (
            user_emb_0.norm().pow(2) +
            pos_emb_0.norm().pow(2) +
            neg_emb_0.norm().pow(2)
    ) / float(len(users))

    # compute BPR loss from user, positive item, and negative item embeddings
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)

    bpr_loss = torch.mean(torch_nnf.softplus(neg_scores - pos_scores))

    return bpr_loss, reg_loss


def compute_metrics(user_embs: torch.Tensor,
                    item_embs: torch.Tensor,
                    n_usr: int,
                    n_itm: int,
                    train_data: pd.DataFrame,
                    test_data: pd.DataFrame,
                    top_k: int,
                    dev: Union[torch.device, str] = "cpu"):
    if isinstance(dev, str):
        dev = torch.device(dev)

    relevance_score = torch.matmul(user_embs, item_embs.T)

    # create dense tensor of all user-item interactions
    connections = torch.stack(
        (
            torch.LongTensor(train_data['user_id_new'].values),
            torch.LongTensor(train_data['item_id_new'].values)
        )
    )
    values = torch.ones((len(train_data),), dtype=torch.float64)
    interactions_t = torch.sparse_coo_tensor(
        connections,
        values,
        (n_usr, n_itm)
    ).to_dense().to(dev)

    # mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    top_k_relevance_indices = torch.topk(relevance_score, top_k).indices
    if dev.type == "cuda":
        top_k_relevance_indices = top_k_relevance_indices.cpu()
    top_k_relevance_indices_df = pd.DataFrame(
        top_k_relevance_indices.numpy(),
        columns=[f'top_idx_{x + 1}' for x in range(top_k)]
    )
    top_k_relevance_indices_df['user_id_new'] = top_k_relevance_indices_df.index
    top_k_relevance_indices_df['top_rvt_item'] = top_k_relevance_indices_df[
        [f'top_idx_{x + 1}' for x in range(top_k)]
    ].values.tolist()
    top_k_relevance_indices_df = top_k_relevance_indices_df[['user_id_new', 'top_rvt_item']]

    test_interacted_items = test_data.groupby('user_id_new')['item_id_new'].apply(list).reset_index()
    metrics_df = pd.merge(
        test_interacted_items,
        top_k_relevance_indices_df,
        how='left',
        on='user_id_new'
    )
    metrics_df['intersection_itm'] = [
        list(set(a).intersection(b))
        for a, b in zip(metrics_df.item_id_new, metrics_df.top_rvt_item)
    ]

    metrics_df['recall'] = metrics_df.apply(lambda x: len(x['intersection_itm']) / len(x['item_id_new']), axis=1)
    metrics_df['precision'] = metrics_df.apply(lambda x: len(x['intersection_itm']) / top_k, axis=1)

    return metrics_df['recall'].mean(), metrics_df['precision'].mean()


class RecModel(nn.Module):
    def __init__(self):
        super(RecModel, self).__init__()

    def encode_minibatch(self,
                         users: torch.Tensor,
                         pos_items: torch.Tensor,
                         neg_items: torch.Tensor,
                         edge_index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raise NotImplemented("Should be implemented in child class")

    def train_and_eval(self,
                       optim: torch.optim.Optimizer,
                       train_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       epochs: int,
                       n_usr: int,
                       n_itm: int,
                       top_k: int,
                       batch_size: int,
                       decay: float,
                       dev: Union[torch.device, str] = "cpu",
                       ckpt_path_recall: Union[str, Path, None] = None,
                       ckpt_path_precision: Union[str, Path, None] = None):
        train_edge_index = get_edge_index(train_data, n_usr, dev)
        loss_list_epoch = []
        bpr_loss_list_epoch = []
        reg_loss_list_epoch = []

        recall_list = []
        precision_list = []

        best_recall = 0
        best_precision = 0

        for epoch in range(epochs):
            n_batch = int(len(train_data) / batch_size)

            # progress bar
            progress_train = tqdm(
                range(1, n_batch + 1),
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                total=n_batch
            )
            final_loss_total = 0
            bpr_loss_total = 0
            reg_loss_total = 0

            # Train
            self.train()
            for batch_idx in progress_train:
                optim.zero_grad()

                users, pos_items, neg_items = get_batch(train_data, batch_size, n_usr, n_itm, dev)
                (
                    users_emb,
                    pos_emb,
                    neg_emb,
                    user_emb_0,
                    pos_emb_0,
                    neg_emb_0
                ) = self.encode_minibatch(
                    users,
                    pos_items,
                    neg_items,
                    train_edge_index
                )

                bpr_loss, reg_loss = compute_bpr_loss(
                    users, users_emb, pos_emb, neg_emb,
                    user_emb_0, pos_emb_0, neg_emb_0
                )

                bpr_loss_total += bpr_loss.item()
                reg_loss_total += reg_loss.item()

                reg_loss = decay * reg_loss
                final_loss = bpr_loss + reg_loss

                final_loss_total += final_loss.item()

                final_loss.backward()
                optim.step()

                progress_train.set_postfix(
                    **{
                        "final loss": final_loss_total / batch_idx,
                        "bpr loss": bpr_loss_total / batch_idx,
                        "reg loss": reg_loss_total / batch_idx,
                    }
                )

            self.eval()
            with torch.no_grad():
                _, out = self(train_edge_index)
                emb_u, emb_i = torch.split(out, [n_usr, n_itm])
                test_top_k_recall, test_top_k_precision = compute_metrics(
                    emb_u, emb_i, n_usr, n_itm, train_data, test_data, top_k, dev
                )

            loss_list_epoch.append(round(final_loss_total / n_batch, 4))
            bpr_loss_list_epoch.append(round(bpr_loss_total / n_batch, 4))
            reg_loss_list_epoch.append(round(reg_loss_total / n_batch, 4))

            recall_list.append(round(test_top_k_recall, 4))
            precision_list.append(round(test_top_k_precision, 4))

            print(
                f"Validation, "
                f"Top K Recall: {test_top_k_recall:.4f}, "
                f"Top K Precision: {test_top_k_precision:.4f}"
            )

            if ckpt_path_recall and best_recall < test_top_k_recall:
                best_recall = test_top_k_recall
                torch.save(self.state_dict(), ckpt_path_recall)
            if ckpt_path_precision and best_precision < test_top_k_precision:
                best_precision = test_top_k_precision
                torch.save(self.state_dict(), ckpt_path_precision)

        return (
            loss_list_epoch,
            bpr_loss_list_epoch,
            reg_loss_list_epoch,
            recall_list,
            precision_list
        )


class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class RecSysGNN(RecModel):
    def __init__(
            self,
            latent_dim: int,
            n_layers: int,
            n_usr: int,
            n_itm: int
    ):
        super(RecSysGNN, self).__init__()
        self.embedding = nn.Embedding(n_usr + n_itm, latent_dim)
        self.conv_s = nn.ModuleList(LightGCNConv() for _ in range(n_layers))
        self.init_parameters()

    def init_parameters(self):
        # Authors of LightGCN report higher results with normal initialization
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb0 = self.embedding.weight
        emb_s = [emb0]

        emb = emb0
        for conv in self.conv_s:
            emb = conv(x=emb, edge_index=edge_index)
            emb_s.append(emb)

        out = torch.mean(torch.stack(emb_s, dim=0), dim=0)

        return emb0, out

    def encode_minibatch(self,
                         users: torch.Tensor,
                         pos_items: torch.Tensor,
                         neg_items: torch.Tensor,
                         edge_index: torch.Tensor) -> tuple[torch.Tensor, ...]:
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items]
        )
