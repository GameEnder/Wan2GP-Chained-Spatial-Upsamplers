"""Chained Spatial Upsamplers Plugin for Wan2GP."""

from .chained_upsampler import ChainedSpatialUpsampler
from .plugin import ConfigTabPlugin, PlugIn_Id, PlugIn_Name

__all__ = ["ChainedSpatialUpsampler", "ConfigTabPlugin", "PlugIn_Id", "PlugIn_Name"]
