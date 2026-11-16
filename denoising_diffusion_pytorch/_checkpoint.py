from os import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def atomic_torch_save(data, path):
    path = Path(path)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir = path.parent, prefix = f'.{path.name}.', suffix = '.tmp', delete = False) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(data, temporary)
        replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok = True)
