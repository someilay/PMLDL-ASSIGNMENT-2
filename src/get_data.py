import argparse
import time
import zipfile

from pathlib import Path
from download_from_url import download_url
from find_root_dir import get_root_path


parser = argparse.ArgumentParser(
    prog='Download data script',
    description='Downloads and unpack data'
)
parser.add_argument('-r', '--reload', help='Reload data', action='store_true')


def get_url(raw_path: Path, url_file: str = 'ml-100k-url'):
    with open(raw_path / url_file, 'r') as f:
        return f.readline().replace('\n', '').lstrip().rstrip()


def download(zip_path: Path, url: str, reload: bool):
    print('Downloading zip...')

    if zip_path.exists() and not reload:
        print('Zip present, skipping...')
        return

    time.sleep(0.25)  # Dummy wait for IO sync
    download_url(url, zip_path)
    print('Download complete')


def unpack(zip_path: Path, unpack_to: Path, reload: bool):
    print(f'Extracting {zip_path.name}')

    if (unpack_to / 'ml-100k').exists() and not reload:
        print('Unzipped data present, skipping...')
        return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unpack_to)
    print('Unpack complete')


def main():
    args = parser.parse_args()
    reload: bool = args.reload

    root_path = get_root_path()
    raw_path = root_path / 'data' / 'raw'
    zip_path = root_path / 'ml-100k.zip'

    data_url = get_url(raw_path)
    download(zip_path, data_url, reload)
    unpack(zip_path, raw_path, reload)


if __name__ == '__main__':
    main()
