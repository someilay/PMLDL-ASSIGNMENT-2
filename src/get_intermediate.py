import argparse
import pandas as pd
import shutil

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


def generate_user_mapping(interim_path: Path, merged_path: Path, reload: bool) -> Path:
    user_map_path = interim_path / 'user_mapping.csv'

    if user_map_path.exists() and not reload:
        print("User map data is already present")
        return user_map_path

    print("Generating user map data")

    merged = pd.read_csv(merged_path, sep="\t")
    user_map = merged.groupby(['user_id', 'user_id_new']).size().reset_index()
    user_map = user_map[['user_id', 'user_id_new']]

    # Save
    user_map.to_csv(user_map_path, sep="\t", index=False)
    print("Done")
    return user_map_path


def generate_item_mapping(interim_path: Path, merged_path: Path, reload: bool) -> Path:
    item_map_path = interim_path / 'item_mapping.csv'

    if item_map_path.exists() and not reload:
        print("Item map data is already present")
        return item_map_path

    print("Generating item map data")

    merged = pd.read_csv(merged_path, sep="\t")
    item_map = merged.groupby(['item_id', 'item_id_new']).size().reset_index()
    item_map = item_map[['item_id', 'item_id_new']]

    # Save
    item_map.to_csv(item_map_path, sep="\t", index=False)
    print("Done")
    return item_map_path


def generate_benchmark_data(benchmark_data_path: Path,
                            ml_path: Path,
                            user_map_path: Path,
                            item_map_path: Path,
                            merged_path: Path,
                            reload: bool):
    print('Generating benchmark data')
    u_data_columns_name = ['user_id', 'item_id', 'rating', 'timestamp']
    u_user_columns = ['user_id', 'age', 'gender', 'occupation', 'zip code']
    u_item_columns = [
        'item_id', 'movie_title', 'release date', 'video release date',
        'IMDb URL', 'unknown', 'Action', 'Adventure',
        'Animation', "Children", 'Comedy', 'Crime',
        'Documentary', 'Drama', 'Fantasy', "Film-Noir",
        'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    u_user = pd.read_csv(ml_path / "u.user", sep="|", names=u_user_columns)
    u_item = pd.read_csv(
        ml_path / "u.item",
        sep="|",
        names=u_item_columns,
        encoding='iso-8859-1'
    )
    user_map = pd.read_csv(user_map_path, sep='\t')
    item_map = pd.read_csv(item_map_path, sep='\t')

    for tp in ['test', 'base']:
        for i in range(1, 6):
            test_data_path = benchmark_data_path / f'merged-{tp}-{i}.csv'

            if test_data_path.exists() and not reload:
                print(f'{test_data_path.name} is already present')
                continue

            u_data = pd.read_csv(ml_path / f"u{i}.{tp}", sep="\t", names=u_data_columns_name)
            merged = pd.merge(u_data, u_user, on='user_id')
            merged = pd.merge(merged, u_item, on='item_id')
            merged = pd.merge(merged, user_map, on='user_id')
            merged = pd.merge(merged, item_map, on='item_id')
            merged.to_csv(test_data_path, sep='\t', index=False)

    all_data_path = benchmark_data_path / merged_path.name

    if not all_data_path.exists():
        shutil.copyfile(merged_path, all_data_path)

    print('Done')


def main():
    args = parser.parse_args()
    reload: bool = args.reload

    root_path = get_root_path()
    data_path = root_path / 'data'

    ml_path = data_path / 'raw' / 'ml-100k'
    interim_path = data_path / 'interim'
    benchmark_data_path = root_path / 'benchmark' / 'data'

    merged_path = generate_merged_data(ml_path, interim_path, reload)
    user_map_path = generate_user_mapping(interim_path, merged_path, reload)
    item_map_path = generate_item_mapping(interim_path, merged_path, reload)
    generate_benchmark_data(benchmark_data_path, ml_path, user_map_path, item_map_path, merged_path, reload)


if __name__ == '__main__':
    main()
