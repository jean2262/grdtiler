"""Top-level package for grdtiling."""

__author__ = """Jean Renaud MIADANA"""
__email__ = "jean-renaud.miadana@ocean-scope.com"
__all__ = ['tiling_prod', 'tiling_by_point', 'tiling_wv', 'make_detrend', 'crosses_antimeridian', 'normalize_longitudes_to_360', 'add_tiles_footprint', 'save_tile']

from .grdtiler import tiling_prod, tiling_by_point, tiling_wv
from .make_detrend import make_detrend
from .antimeridian import crosses_antimeridian, normalize_longitudes_to_360
from .tools import add_tiles_footprint, save_tile
