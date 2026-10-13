"""Shared uint8 RGB preprocessing; encoder-specific normalization stays outside."""

from .processor import ImageProfile, ProcessedWindow, SharedImageProcessor, load_profile

__all__ = ["ImageProfile", "ProcessedWindow", "SharedImageProcessor", "load_profile"]
__version__ = "1.0.0"
