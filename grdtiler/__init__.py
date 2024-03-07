"""Top-level package for grdtiling."""

__author__ = """Jean Renaud MIADANA"""
__email__ = "jrenaud495@gmail.com"
__version__ = "0.0.1"
__all__ = ['tiling_prod', 'tiling_by_point', 'tiles_images', 'get_tiles_footprint', 'plot_cartopy_data']

from .grdtiler import *
from .tools import *
