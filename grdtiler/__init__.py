"""Top-level package for grdtiling."""

__author__ = """Jean Renaud MIADANA"""
__email__ = "jrenaud495@gmail.com"
__all__ = ['tiling_prod', 'tiling_by_point', 'tiling_wv']

from .grdtiler import tiling_prod, tiling_by_point, tiling_wv
from .tools import add_tiles_footprint, save_tile
