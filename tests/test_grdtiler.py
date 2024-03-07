#!/usr/bin/env python

"""Tests for `grdtiling` package."""

import grdtiler
import numpy as np
import pytest
import xsar


@pytest.fixture
def path_to_product_sample():
    filename = xsar.get_test_file('S1A_IW_GRDH_1SDV_20210909T130650_20210909T130715_039605_04AE83_Z010.SAFE')
    return filename


def test_tile_comparison(path_to_product_sample):
    # Generate tiles using grdtiling.tiling_prod
    ds_x, tiles_x = grdtiler.tiling_prod(path=path_to_product_sample, nperseg={'line': 100, 'sample': 100},
                                         resolution='1000m', tiling_mod='xtiling', centering=True, side='left',
                                         noverlap=0, save_tiles=False)

    # Generate tiles using tiling_prod
    ds_t, tiles_t = grdtiler.tiling_prod(path=path_to_product_sample, nperseg={'line': 100, 'sample': 100},
                                         resolution='1000m', tiling_mod='tiling', centering=True, side='left',
                                         noverlap=0, save_tiles=False)

    # Comparison
    for i in range(len(tiles_x)):
        assert np.array_equal(tiles_x[i].sel(pol='VV').sigma0.values,
                              tiles_t[i].sel(pol='VV').sigma0.values), f"Tile {i} values are not equal"
