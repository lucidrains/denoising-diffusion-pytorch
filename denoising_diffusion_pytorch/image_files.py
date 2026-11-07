from pathlib import Path


def image_files(folder, exts):
    suffixes = {'.' + ext.lstrip('.').lower() for ext in exts}
    return sorted(
        path for path in Path(folder).rglob('*')
        if path.suffix.lower() in suffixes and path.is_file()
    )
