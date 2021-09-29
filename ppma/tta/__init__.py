from . import aliases
from .base import Compose
from .transforms import (
    Add,
    FiveCrops,
    HorizontalFlip,
    Multiply,
    Resize,
    Rotate90,
    Scale,
    VerticalFlip,
)
from .wrappers import ClasTTA
