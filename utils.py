import os


def file_relative_path(dunderfile, rel_path: str) -> str:
    return os.path.join(os.path.dirname(dunderfile), rel_path)
