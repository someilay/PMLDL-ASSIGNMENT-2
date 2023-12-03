import argparse
import time
import zipfile

from pathlib import Path
from download_from_url import download_url
from find_root_dir import get_root_path

# Define a parser for command-line arguments
parser = argparse.ArgumentParser(
    prog='Download data script',
    description='Downloads and unpack data'
)
parser.add_argument('-r', '--reload', help='Reload data', action='store_true')


def get_url(raw_path: Path, url_file: str = 'ml-100k-url'):
    """Reads the dataset URL from a file.

    Args:
        raw_path (Path): The path to the raw data directory.
        url_file (str): The filename containing the dataset URL.

    Returns:
        str: The dataset URL.
    """
    with open(raw_path / url_file, 'r') as f:
        return f.readline().replace('\n', '').lstrip().rstrip()


def download(zip_path: Path, url: str, reload: bool):
    """
    Downloads the dataset zip file from the provided URL.

    Args:
        zip_path (Path): The path where the dataset zip file will be saved.
        url (str): The URL of the dataset zip file.
        reload (bool): If True, force download even if the file exists.
    """
    print('Downloading zip...')

    if zip_path.exists() and not reload:
        print('Zip present, skipping...')
        return

    time.sleep(0.25)  # Dummy wait for IO sync
    download_url(url, zip_path)
    print('Download complete')


def unpack(zip_path: Path, unpack_to: Path, reload: bool):
    """Unpacks the dataset zip file.

    Args:
        zip_path (Path): The path to the dataset zip file.
        unpack_to (Path): The directory where the dataset will be unpacked.
        reload (bool): If True, force extraction even if the directory exists.
    """
    print(f'Extracting {zip_path.name}')

    if (unpack_to / 'ml-100k').exists() and not reload:
        print('Unzipped data present, skipping...')
        return

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unpack_to)
    print('Unpack complete')


def main():
    """Main function to execute the data download and extraction process."""
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
