from pathlib import Path


def get_root_path() -> Path:
    """
    Get the root path of the project.

    Returns:
        Path: Path to the project's root directory.
    """
    candidates = ['../..', '..', '.']
    for candidate in candidates:
        readme_candidate = Path(candidate) / 'README.md'
        readme_candidate = readme_candidate.resolve()
        if not readme_candidate.exists():
            continue
        with open(readme_candidate) as readme:
            if not readme.readline().startswith('## PMLDL ASSIGNMENT 2'):
                continue
            return Path(candidate).resolve()

    raise FileNotFoundError(f'Cannot define a root path(')
