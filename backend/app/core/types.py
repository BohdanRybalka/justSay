from enum import Enum


class ProviderMode(str, Enum):
    CLOUD = "cloud"
    LOCAL = "local"
