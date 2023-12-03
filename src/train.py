print("Importing stuff...")

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import torch

from typing import Union
from pathlib import Path

from sklearn.model_selection import train_test_split

from models_and_metrics import RecSysGNN, FeaturedRecSysGNN
from find_root_dir import get_root_path
from utils import get_item_features, get_user_features, load_merged_data, count_user_items
from config import (
    ModelTypes, MODEL2HISTORY, MODEL2RECALL_SAVE_PATH, MODEL2PRECISION_SAVE_PATH,
    BEST_RECALL_MODEL_PATH, BEST_PRECISION_MODEL_PATH, METRICS_HISTORY_PATH
)

# Command-line arguments
parser = argparse.ArgumentParser(
    prog='Train RecSys script',
    description='Trains model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('-e', '--epochs', help='Number of epochs', type=int, default=75)
parser.add_argument('-b', '--batch-size', help='Batch size', type=int, default=1024)
parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, default=1e-3)
parser.add_argument('-rs', '--random-state', help='Random state', type=int, default=42)
parser.add_argument(
    '-tr', '--test-ration',
    help='Test ration (should be in range [0.05 - 0.5])',
    type=float,
    default=0.2,
)
parser.add_argument('-d', '--device', help='Device for training (cuda or cpu)', type=str, default='cuda')
parser.add_argument(
    '-dec', '--decay',
    help='Decay regularization, positive, less than 0.1',
    type=float,
    default=0.0001,
)
parser.add_argument(
    '-k', '--top-k',
    help='Top k recommendations will used for validation, should be positive',
    type=int,
    default=20
)
parser.add_argument(
    '-rbs', '--recall-best-save',
    help='Path to folder where best model will be saved (by recall score)',
    type=str,
    default=BEST_RECALL_MODEL_PATH.as_posix(),
)
parser.add_argument(
    '-pbs', '--precision-best-save',
    help='Path to folder where best model will be saved (by precision score)',
    type=str,
    default=BEST_PRECISION_MODEL_PATH.as_posix(),
)
parser.add_argument(
    '-hs', '--history',
    help='Path to folder where metrics history will be saved',
    type=str,
    default=METRICS_HISTORY_PATH.as_posix(),
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


def get_train_val_split(data: pd.DataFrame, random_state: int, test_ration: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and validation sets.

    Parameters:
        data (pd.DataFrame): Merged DataFrame containing user-item interactions.
        random_state (int): Random seed for reproducibility.
        test_ration (float): Ratio of data to be used for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and validation DataFrames.
    """
    data = data.copy()
    data['id'] = range(len(data))
    test_size = int(test_ration * len(data))

    # Save all unique users
    all_no_user_cols = [i for i in data.columns if i != 'user_id']
    first_user_entries = data.groupby('user_id')[all_no_user_cols].first().reset_index()
    first_user_entries = first_user_entries[data.columns]

    data = data[
        ~data['id'].isin(first_user_entries['id'].values)
    ]

    # Save all unique movies
    all_no_item_cols = [i for i in data.columns if i != 'item_id']
    first_movie_entries = data.groupby('item_id')[all_no_item_cols].first().reset_index()
    first_movie_entries = first_movie_entries[data.columns]

    data = data[
        ~data['id'].isin(first_movie_entries['id'].values)
    ]

    # Split data
    train, test = train_test_split(
        data.values,
        test_size=test_size,
        random_state=random_state
    )
    train_data = pd.DataFrame(train, columns=data.columns)
    train_data = pd.concat([train_data, first_user_entries, first_movie_entries], axis=0)
    test_data = pd.DataFrame(test, columns=data.columns)

    train_data = train_data.drop('id', axis=1)
    test_data = test_data.drop('id', axis=1)
    data = data.drop('id', axis=1)

    # Restore types
    for col in data.columns:
        train_data[col] = train_data[col].astype(data[col].dtype)
        test_data[col] = test_data[col].astype(data[col].dtype)

    return train_data, test_data


def get_dataframes(merged_data: pd.DataFrame,
                   random_state: int,
                   test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and testing DataFrames from merged data.

    This function filters the merged data based on the 'rating' column, splits it into training and testing sets,
    and ensures the correctness of the split.

    Parameters:
        merged_data (pd.DataFrame): Merged DataFrame containing user-item interactions.
        random_state (int): Random seed for reproducibility.
        test_ratio (float): Ratio of data to be used for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames.
    """
    # Filter data based on the 'rating' column
    merged_data_good = merged_data[merged_data['rating'] > 2]
    train_df, test_df = get_train_val_split(merged_data_good, random_state, test_ratio)
    # Ensure correctness of the split
    assert len(train_df) + len(test_df) == len(merged_data_good)
    return train_df, test_df


def plot_metrics(metrics: list,
                 y_label: str,
                 title: str,
                 epochs: int,
                 step: int,
                 save_to: Path):
    """
    Plot metrics over epochs and save the plot.

    This function takes a list of metrics, along with labels and parameters for the plot, and creates a line plot
    representing the metrics over epochs. The plot is then saved to the specified location.

    Parameters:
        metrics (list): List of metric values to be plotted.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        epochs (int): Total number of epochs.
        step (int): Interval between x-axis ticks.
        save_to (Path): Path to save the plot.
    """
    epochs_range = range(1, epochs + 1, 1)
    x_tics = list(range(step, epochs + 1, step))
    if step > 1:
        x_tics = [1] + x_tics

    plt.clf()
    plt.figure(figsize=(16, 4))
    plt.title(title)
    plt.plot(
        epochs_range,
        metrics,
        label=y_label,
    )
    plt.legend(loc="best")
    plt.xlabel("Epochs")
    plt.xticks(x_tics)
    plt.ylabel(y_label)
    plt.savefig(save_to / (title.lower().replace(' ', '_') + '.png'))


def main():
    """
    Train and evaluate a recommendation system model based on the provided command-line arguments.

    This function parses command-line arguments, checks their validity, initializes the model and optimizer,
    trains and evaluates the model, and saves the training history and best models.
    """
    # Parse arguments
    args = parser.parse_args()
    epochs = args.epochs
    batch_size: int = args.batch_size
    lr: float = args.learning_rate
    random_state: int = args.random_state
    test_ration: float = args.test_ration
    device: Union[torch.device, str] = args.device
    decay: float = args.decay
    top_k: int = args.top_k
    recall_best_save: Union[str, Path] = args.recall_best_save
    precision_best_save: Union[str, Path] = args.precision_best_save
    history: Union[str, Path] = args.history
    emb_dim: int = args.emb_dim
    n_layers: int = args.n_layers
    model_type: str = args.model_type

    # Convert paths to Path objects
    recall_best_save = Path(recall_best_save)
    precision_best_save = Path(precision_best_save)
    history = Path(history)

    # Check params
    if not 0.05 <= test_ration <= 0.5:
        print('test-ration should be in range [0.05, 0.5]')
        exit(1)
    if device not in ['cpu', 'cuda']:
        print('device should be cpu or cuda')
        exit(1)
    if device == 'cuda' and not torch.cuda.is_available():
        print("Cuda unavailable, switching to cpu")
        device = "cpu"
    if not 0 < decay <= 0.1:
        print('decay should be in range (0, 0.1]')
        exit(1)
    if top_k <= 0:
        print('top-k should be positive integer')
        exit(1)
    if recall_best_save != BEST_RECALL_MODEL_PATH and not recall_best_save.parent.exists():
        print(f'Can not find "{recall_best_save.parent}"')
        exit(1)
    if precision_best_save != BEST_PRECISION_MODEL_PATH and not precision_best_save.parent.exists():
        print(f'Can not find "{precision_best_save.parent}"')
        exit(1)
    if history != METRICS_HISTORY_PATH and not history.exists():
        print(f'Can not find "{history}"')
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

    # Set random seed and device
    torch.manual_seed(random_state)
    device = torch.device(device)

    # Define paths
    root_path = get_root_path()
    data_path = root_path / 'data'
    merged_path = data_path / 'interim' / 'merged.csv'
    # Adjust paths if using default paths
    if recall_best_save == BEST_RECALL_MODEL_PATH:
        recall_best_save = root_path / recall_best_save
    if precision_best_save == BEST_PRECISION_MODEL_PATH:
        precision_best_save = root_path / precision_best_save
    if history == METRICS_HISTORY_PATH:
        history = root_path / history

    # Append model-specific paths
    recall_best_save = recall_best_save / MODEL2RECALL_SAVE_PATH[model_type]
    precision_best_save = precision_best_save / MODEL2PRECISION_SAVE_PATH[model_type]
    history = history / MODEL2HISTORY[model_type]
    history.mkdir(exist_ok=True)

    # Load data
    merged_data = load_merged_data(merged_path)
    merged_data_good = merged_data[merged_data['rating'] > 2]
    train_df, test_df = get_train_val_split(merged_data_good, random_state, test_ration)
    n_users, n_items = count_user_items(merged_data)
    movie_cat_columns = list(merged_data.columns[13:-2])

    # Initialize model based on model type
    if model_type == ModelTypes.WITH:
        model = FeaturedRecSysGNN(
            latent_dim=emb_dim,
            n_layers=n_layers,
            n_usr=n_users,
            n_itm=n_items,
            user_features=get_user_features(merged_data, device),
            item_features=get_item_features(merged_data, movie_cat_columns, device),
        ).to(device)
    else:
        model = RecSysGNN(
            latent_dim=emb_dim,
            n_layers=n_layers,
            n_usr=n_users,
            n_itm=n_items,
        ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    light_loss, light_bpr, light_reg, light_recall, light_precision = model.train_and_eval(
        optim=optimizer,
        train_data=train_df,
        test_data=test_df,
        epochs=epochs,
        n_usr=n_users,
        n_itm=n_items,
        top_k=top_k,
        batch_size=batch_size,
        decay=decay,
        dev=device,
        ckpt_path_recall=recall_best_save,
        ckpt_path_precision=precision_best_save,
    )

    # Train and evaluate the model
    step = max(epochs // 10, 1)
    plot_metrics(light_loss, "Final Loss", "Final loss per epoch", epochs, step, history)
    plot_metrics(light_bpr, "BPR Loss", "BPR loss per epoch", epochs, step, history)
    plot_metrics(light_reg, "Reg Loss", "Reg loss per epoch", epochs, step, history)
    plot_metrics(light_recall, "Recall", "Recall per epoch", epochs, step, history)
    plot_metrics(light_precision, "Precision", "Precision per epoch", epochs, step, history)


if __name__ == '__main__':
    main()
