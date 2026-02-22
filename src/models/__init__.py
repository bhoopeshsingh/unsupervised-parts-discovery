"""DINO model wrappers"""

from .dino_extractor import DinoExtractor
from .dino_finetuner import DinoSemanticFinetuner

__all__ = ["DinoExtractor", "DinoSemanticFinetuner"]
