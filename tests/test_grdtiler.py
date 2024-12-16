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
    # Load xsarslc pregenerated tiles
    xtiling_tiles = np.load("tests/xtiling_tiles.npy", allow_pickle=True)

    # Generate tiles using tiling_prod
    _, tiles_t = grdtiler.tiling_prod(path=path_to_product_sample, tile_size={'line': 17600, 'sample': 17600},
                                         resolution='400m', centering=True, side='left',
                                         noverlap=0, save=False)

    # Comparison
    for i, tile_x in enumerate(xtiling_tiles):
        assert np.array_equal(tile_x, tiles_t.sel(tile=i, pol='VV').sigma0.values), f"Tile {i} values are not equal"
