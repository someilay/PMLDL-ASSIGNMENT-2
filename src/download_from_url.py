import urllib.request

from pathlib import Path
from typing import Optional
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, t_size: Optional[int] = None):
        if t_size is not None:
            self.total = t_size
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
