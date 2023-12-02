import argparse
import pandas as pd

from pathlib import Path
from sklearn import preprocessing as pp
from find_root_dir import get_root_path


parser = argparse.ArgumentParser(
    prog='Generate intermediate data script',
    description='Generates intermediate data and places to necessary folders'
)
parser.add_argument('-r', '--reload', help='Regenerate data', action='store_true')


def generate_merged_data(ml_path: Path, interim_path: Path, reload: bool) -> Path:
    merged_path = interim_path / "merged.csv"

    if merged_path.exists() and not reload:
        print("Merged data is already present")
        return merged_path

    print("Generating merged data")
    u_data_columns_name = ['user_id', 'item_id', 'rating', 'timestamp']
    u_data = pd.read_csv(ml_path / "u.data", sep="\t", names=u_data_columns_name)
    u_data['timestamp'] = pd.to_datetime(u_data['timestamp'], unit='s')

    u_user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip code']
    u_user = pd.read_csv(ml_path / "u.user", sep="|", names=u_user_columns)

    u_item_columns = [
        'item_id', 'movie_title', 'release date', 'video release date',
        'IMDb URL', 'unknown', 'Action', 'Adventure',
        'Animation', "Children", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', "Film-Noir",
        'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]
    u_item = pd.read_csv(
        ml_path / "u.item",
        sep="|",
        names=u_item_columns,
        encoding='iso-8859-1'
    )

    merged = pd.merge(u_data, u_user, on='user_id')
    merged = pd.merge(merged, u_item, on='item_id')

    le_user = pp.LabelEncoder()
    le_item = pp.LabelEncoder()

    merged['user_id_new'] = le_user.fit_transform(merged['user_id'].values)
    merged['item_id_new'] = le_item.fit_transform(merged['item_id'].values)

    # Save
    merged.to_csv(merged_path, sep="\t", index=False)
    print("Done")
    return merged_path


def generate_user_mapping(interim_path: Path, merged_path: Path, reload: bool):
    user_map_path = interim_path / 'user_mapping.csv'

    if user_map_path.exists() and not reload:
        print("User map data is already present")
        return

    print("Generating user map data")

    merged = pd.read_csv(merged_path, sep="\t")
    user_map = merged.groupby(['user_id', 'user_id_new']).size().reset_index()
    user_map = user_map[['user_id', 'user_id_new']]

    # Save
    user_map.to_csv(user_map_path, sep="\t", index=False)
    print("Done")


def generate_item_mapping(interim_path: Path, merged_path: Path, reload: bool):
    item_map_path = interim_path / 'item_mapping.csv'

    if item_map_path.exists() and not reload:
        print("Item map data is already present")
        return

    print("Generating item map data")

    merged = pd.read_csv(merged_path, sep="\t")
    item_map = merged.groupby(['item_id', 'item_id_new']).size().reset_index()
    item_map = item_map[['item_id', 'item_id_new']]

    # Save
    item_map.to_csv(item_map_path, sep="\t", index=False)
    print("Done")


def main():
    args = parser.parse_args()
    reload: bool = args.reload

    root_path = get_root_path()
    data_path = root_path / 'data'

    ml_path = data_path / 'raw' / 'ml-100k'
    interim_path = data_path / 'interim'

    merged_path = generate_merged_data(ml_path, interim_path, reload)
    generate_user_mapping(interim_path, merged_path, reload)
    generate_item_mapping(interim_path, merged_path, reload)


if __name__ == '__main__':
    main()
