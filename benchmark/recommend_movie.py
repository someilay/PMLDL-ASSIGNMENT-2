print("Importing stuff...")

import argparse

import torch
from typing import Union
from pathlib import Path

from src.find_root_dir import get_root_path
from src.models_and_metrics import compute_metrics, get_edge_index, FeaturedRecSysGNN, RecSysGNN
from src.config import (
    ModelTypes, ModelMetrics, MODEL2RECALL_SAVE_PATH, MODEL2PRECISION_SAVE_PATH,
    BEST_RECALL_MODEL_PATH, BEST_PRECISION_MODEL_PATH
)
from src.utils import load_merged_data, count_user_items, get_user_features, get_item_features

parser = argparse.ArgumentParser(
    prog='Recommend movie script',
    description='Recommend movie',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '-ui', '--user_id',
    help='A user id (REQUIRED!)',
    type=int,
    required=True
)
parser.add_argument(
    '-vd', '--viewed',
    help='Already viewed movies (ids)',
    nargs="*",
    type=int,
)
parser.add_argument('-d', '--device', help='Device for training (cuda or cpu)', type=str, default='cuda')
parser.add_argument(
    '-k', '--top-k',
    help='Top k recommendations will used for validation, should be positive',
    type=int,
    default=20
)
parser.add_argument(
    '-rbs', '--recall-best-save',
    help='Path to folder where best model saved (by recall score)',
    type=str,
    default=BEST_RECALL_MODEL_PATH.as_posix(),
)
parser.add_argument(
    '-pbs', '--precision-best-save',
    help='Path to folder where best model saved (by precision score)',
    type=str,
    default=BEST_PRECISION_MODEL_PATH.as_posix(),
)
parser.add_argument('-ed', '--emb-dim', help='Embedding dimension, positive integer', type=int, default=64)
parser.add_argument(
    '-nl', '--n-layers',
    help='Amount of layers, positive integer',
    type=int,
    default=4
)
parser.add_argument(
    '-mt', '--model-type',
    help=f'Type of model ({ModelTypes.WITHOUT} or {ModelTypes.WITH})',
    type=str,
    default=ModelTypes.WITHOUT
)
parser.add_argument(
    '-mc', '--metric',
    help=f'Loads best model according to metric ({ModelMetrics.RECALL} or {ModelMetrics.PRECISION})',
    type=str,
    default=ModelMetrics.RECALL
)


def main():
    """
    Main function for recommending the movie.
    """
    # Parse arguments
    args = parser.parse_args()
    device: Union[torch.device, str] = args.device
    top_k: int = args.top_k
    recall_best_save: Union[str, Path] = args.recall_best_save
    precision_best_save: Union[str, Path] = args.precision_best_save
    emb_dim: int = args.emb_dim
    n_layers: int = args.n_layers
    model_type: str = args.model_type
    metric: str = args.metric
    viewed: list[int] = args.viewed or []
    user_id: int = args.user_id

    recall_best_save = Path(recall_best_save)
    precision_best_save = Path(precision_best_save)

    # Check params
    if device not in ['cpu', 'cuda']:
        print('device should be cpu or cuda')
        exit(1)
    if device == 'cuda' and not torch.cuda.is_available():
        print("Cuda unavailable, switching to cpu")
        device = "cpu"
    if top_k <= 0:
        print('top-k should be positive integer')
        exit(1)
    if recall_best_save != BEST_RECALL_MODEL_PATH and not recall_best_save.parent.exists():
        print(f'Can not find "{recall_best_save.parent}"')
        exit(1)
    if precision_best_save != BEST_PRECISION_MODEL_PATH and not precision_best_save.parent.exists():
        print(f'Can not find "{precision_best_save.parent}"')
        exit(1)
    if emb_dim <= 0:
        print('emb-dim should be positive integer')
        exit(1)
    if n_layers <= 0:
        print('n-layers should be positive integer')
        exit(1)
    if model_type not in [ModelTypes.WITHOUT, ModelTypes.WITH]:
        print(f'model type should be one of [{ModelTypes.WITHOUT}, {ModelTypes.WITH}]')
        exit(1)
    if metric not in [ModelMetrics.RECALL, ModelMetrics.PRECISION]:
        print(f'metric should be one of [{ModelMetrics.RECALL}, {ModelMetrics.PRECISION}]')
        exit(1)

    # Define paths
    root_path = get_root_path()
    data_path = root_path / 'benchmark' / 'data'
    merged_path = data_path / 'merged.csv'
    user_map_path = root_path / 'data' / 'interim' / 'user_mapping.csv'
    item_map_path = root_path / 'data' / 'interim' / 'item_mapping.csv'

    if recall_best_save == BEST_RECALL_MODEL_PATH:
        recall_best_save = root_path / recall_best_save
    if precision_best_save == BEST_PRECISION_MODEL_PATH:
        precision_best_save = root_path / precision_best_save
    recall_best_save = recall_best_save / MODEL2RECALL_SAVE_PATH[model_type]
    precision_best_save = precision_best_save / MODEL2PRECISION_SAVE_PATH[model_type]
    load_path = recall_best_save if metric == ModelMetrics.RECALL else precision_best_save

    # Load data
    print('Loading the data')
    merged_data = load_merged_data(merged_path)
    item_map = load_merged_data(item_map_path)
    user_map = load_merged_data(user_map_path)
    viewed_mapped = item_map[item_map['item_id'].isin(viewed)].item_id_new.tolist()
    user_id_new = user_map[user_map['user_id'] == user_id].user_id_new.values[0]

    # Define other parameters
    device = torch.device(device)
    n_users, n_items = count_user_items(merged_data)
    movie_cat_columns = list(merged_data.columns[13:-2])

    print('Loading the model')
    if model_type == ModelTypes.WITH:
        model = FeaturedRecSysGNN(
            latent_dim=emb_dim,
            n_layers=n_layers,
            n_usr=n_users,
            n_itm=n_items,
            user_features=get_user_features(merged_data, device),
            item_features=get_item_features(merged_data, movie_cat_columns, device),
        )
        model.load_state_dict(torch.load(load_path, map_location=device))
        model = model.to(device)
    else:
        model = RecSysGNN(
            latent_dim=emb_dim,
            n_layers=n_layers,
            n_usr=n_users,
            n_itm=n_items,
        )
        model.load_state_dict(torch.load(load_path, map_location=device))
        model = model.to(device)

    recommend_id_new = model.predict_for_user(
        user_id_new, viewed_mapped, n_users, n_items, get_edge_index(merged_data, n_users, device)
    )
    recommend_id = item_map[item_map['item_id_new'] == recommend_id_new].item_id.values[0]
    movie_title = merged_data[merged_data['item_id'] == recommend_id]['movie_title'].values[0]
    mean_rating = merged_data[merged_data['item_id'] == recommend_id]['rating'].mean()

    print()
    print('Recommendation id:', recommend_id)
    print('Recommendation title:', movie_title)
    print(f'Recommendation mean rating: {mean_rating:.3f}')


if __name__ == '__main__':
    main()
