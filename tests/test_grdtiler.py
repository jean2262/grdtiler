#!/usr/bin/env python

"""Tests for `grdtiling` package."""

import grdtiler
import numpy as np
import pytest
import xsar
from xsarslc.tools import xtiling, get_tiles


@pytest.fixture
def path_to_product_sample():
    filename = xsar.get_test_file('S1A_IW_GRDH_1SDV_20210909T130650_20210909T130715_039605_04AE83_Z010.SAFE')
    return filename


def test_tile_comparison(path_to_product_sample):
    # Generate tiles using xsarslc
    dataset = xsar.open_dataset(path_to_product_sample, resolution='400m')
    tiles_index = xtiling(ds=dataset, nperseg={'line': 44, 'sample': 44}, noverlap=0, centering=True, side='left')
    tiles_x = get_tiles(ds=dataset, tiles_index=tiles_index)

    # Generate tiles using tiling_prod
    ds_t, tiles_t = grdtiler.tiling_prod(path=path_to_product_sample, tile_size={'line': 17600, 'sample': 17600},
                                         resolution='400m', centering=True, side='left',
                                         noverlap=0, save=False)

    # Comparison
    for i in range(len(tiles_x)):
        assert np.array_equal(tiles_x[i].sel(pol='VV').sigma0.values, tiles_t.sel(tile=i, pol='VV').sigma0.values), f"Tile {i} values are not equal"
