from importlib import metadata

DIST_NAME = "langchain-oxylabs"

try:
    # `__package__` is empty when this module is loaded as a standalone file
    # (for example by scripts/check_imports.py), which would make
    # `metadata.version` raise ValueError rather than PackageNotFoundError.
    __version__ = metadata.version(__package__ or DIST_NAME)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)
