from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("justsay-backend")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
