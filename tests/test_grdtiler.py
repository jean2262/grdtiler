#!/usr/bin/env python

"""Tests for `grdtiling` package."""

import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon

import grdtiler
from grdtiler.grdtiler import (
    _detect_mission,
    crosses_antimeridian,
    find_pixel,
    load_dataset,
    normalize_longitudes_to_360,
    tiling,
)
from grdtiler.tools import add_tiles_footprint, process_single_tile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n_lines=100, n_samples=80, pols=("VV", "VH")):
    """Minimal SAR-like xarray Dataset for unit tests (no real SAR file needed)."""
    rng = np.random.default_rng(0)
    lines = np.arange(n_lines)
    samples = np.arange(n_samples)

    lon = rng.uniform(-5.0, 5.0, (n_lines, n_samples))
    lat = rng.uniform(40.0, 50.0, (n_lines, n_samples))
    sigma0 = rng.uniform(0.0, 1.0, (len(pols), n_lines, n_samples))

    return xr.Dataset(
        {
            "longitude": (["line", "sample"], lon),
            "latitude":  (["line", "sample"], lat),
            "sigma0":    (["pol", "line", "sample"], sigma0),
        },
        coords={"line": lines, "sample": samples, "pol": list(pols)},
    )


def make_tile(n=20, pols=("VV", "VH")):
    """Synthetic tile dataset (tile_line / tile_sample dims, for tools tests)."""
    rng = np.random.default_rng(1)
    tile_lines = np.arange(n)
    tile_samples = np.arange(n)

    lon = rng.uniform(-1.0, 1.0, (n, n))
    lat = rng.uniform(44.0, 46.0, (n, n))

    return xr.Dataset(
        {
            "longitude": (["tile_line", "tile_sample"], lon),
            "latitude":  (["tile_line", "tile_sample"], lat),
        },
        coords={"tile_line": tile_lines, "tile_sample": tile_samples},
    )


# ---------------------------------------------------------------------------
# _detect_mission
# ---------------------------------------------------------------------------

class TestDetectMission:
    def test_sentinel1(self):
        assert _detect_mission("S1A_IW_GRDH_foo") == "S1"

    def test_sentinel1_lowercase(self):
        assert _detect_mission("s1b_ew_grdm_foo") == "S1"

    def test_radarsat2(self):
        assert _detect_mission("RS2_20200101_foo") == "RS2"

    def test_rcm(self):
        assert _detect_mission("RCM1_foo") == "RCM"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported mission"):
            _detect_mission("UNKNOWN_sensor_foo")


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_passthrough_xarray(self):
        ds = make_synthetic_dataset()
        result = load_dataset(ds)
        assert result is ds

    def test_unsupported_path_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset("/data/ENVISAT_product.N1")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            load_dataset("/data/some_image.tif")


# ---------------------------------------------------------------------------
# tiling()
# ---------------------------------------------------------------------------

class TestTiling:
    def test_basic_no_centering(self):
        ds = make_synthetic_dataset(n_lines=100, n_samples=80)
        tiles = tiling(ds, tile_size=20, noverlap=0, centering=False,
                       side="left", add_footprint=False)
        assert "tile" in tiles.dims
        assert tiles.sizes["tile"] == (100 // 20) * (80 // 20)  # 5 * 4 = 20

    def test_tile_dimensions(self):
        ds = make_synthetic_dataset(n_lines=60, n_samples=60)
        tiles = tiling(ds, tile_size=20, noverlap=0, centering=False,
                       side="left", add_footprint=False)
        assert tiles.sizes["tile_line"] == 20
        assert tiles.sizes["tile_sample"] == 20

    def test_centering_produces_fewer_or_equal_tiles(self):
        ds = make_synthetic_dataset(n_lines=105, n_samples=85)
        tiles_no_center = tiling(ds, tile_size=20, noverlap=0,
                                 centering=False, side="left", add_footprint=False)
        tiles_center = tiling(ds, tile_size=20, noverlap=0,
                              centering=True, side="left", add_footprint=False)
        assert tiles_center.sizes["tile"] <= tiles_no_center.sizes["tile"]

    def test_overlap_reduces_step(self):
        ds = make_synthetic_dataset(n_lines=100, n_samples=100)
        tiles_no_ov = tiling(ds, tile_size=20, noverlap=0,  centering=False,
                             side="left", add_footprint=False)
        tiles_ov    = tiling(ds, tile_size=20, noverlap=10, centering=False,
                             side="left", add_footprint=False)
        assert tiles_ov.sizes["tile"] > tiles_no_ov.sizes["tile"]

    def test_dict_tile_size(self):
        ds = make_synthetic_dataset(n_lines=60, n_samples=80)
        tiles = tiling(ds, tile_size={"line": 20, "sample": 40},
                       noverlap=0, centering=False, side="left", add_footprint=False)
        assert tiles.sizes["tile_line"] == 20
        assert tiles.sizes["tile_sample"] == 40

    def test_dict_noverlap(self):
        ds = make_synthetic_dataset(n_lines=60, n_samples=60)
        # Should not raise with dict noverlap (previous bug: TypeError)
        tiles = tiling(ds, tile_size=20, noverlap={"line": 5, "sample": 5},
                       centering=False, side="left", add_footprint=False)
        assert tiles.sizes["tile"] > 0

    def test_noverlap_too_large_raises(self):
        ds = make_synthetic_dataset(n_lines=60, n_samples=60)
        with pytest.raises(ValueError, match="Overlap size must be less than tile size"):
            tiling(ds, tile_size=10, noverlap=10, centering=False,
                   side="left", add_footprint=False)

    def test_dataset_too_small_raises(self):
        ds = make_synthetic_dataset(n_lines=5, n_samples=5)
        with pytest.raises(ValueError, match="No tiles generated"):
            tiling(ds, tile_size=20, noverlap=0, centering=False,
                   side="left", add_footprint=False)

    def test_add_footprint(self):
        ds = make_synthetic_dataset(n_lines=60, n_samples=60)
        tiles = tiling(ds, tile_size=20, noverlap=0, centering=False,
                       side="left", add_footprint=True)
        assert "tile_footprint" in tiles
        assert "lon_centroid" in tiles
        assert "lat_centroid" in tiles


# ---------------------------------------------------------------------------
# find_pixel
# ---------------------------------------------------------------------------

class TestFindPixel:
    def _make_grid(self, n_lines=200, n_samples=150):
        """Regular lon/lat grid centred around (2, 45)."""
        lines   = np.arange(n_lines,   dtype=float)
        samples = np.arange(n_samples, dtype=float)
        lon = np.linspace(0.0, 4.0, n_samples)
        lat = np.linspace(43.0, 47.0, n_lines)
        lon2d, lat2d = np.meshgrid(lon, lat)
        return xr.Dataset(
            {"longitude": (["line", "sample"], lon2d),
             "latitude":  (["line", "sample"], lat2d)},
            coords={"line": lines, "sample": samples},
        )

    def test_exact_pixel_recovered(self):
        ds = self._make_grid()
        # Pick the pixel at (line=100, sample=75) — its exact lon/lat
        lon0 = float(ds.longitude.isel(line=100, sample=75))
        lat0 = float(ds.latitude.isel(line=100, sample=75))
        line, sample = find_pixel(ds, lon0, lat0)
        assert line == 100.0
        assert sample == 75.0

    def test_nearest_pixel_for_off_grid_point(self):
        ds = self._make_grid()
        # Point slightly off-grid — should return the nearest pixel
        lon0 = float(ds.longitude.isel(line=50, sample=30)) + 1e-6
        lat0 = float(ds.latitude.isel(line=50, sample=30)) + 1e-6
        line, sample = find_pixel(ds, lon0, lat0)
        assert abs(line   - 50.0) <= 1.0
        assert abs(sample - 30.0) <= 1.0

    def test_large_grid_performance(self):
        """Coarse-to-fine search must complete quickly on a large grid."""
        import time
        n = 1000
        lines   = np.arange(n, dtype=float)
        samples = np.arange(n, dtype=float)
        lon2d, lat2d = np.meshgrid(np.linspace(0, 10, n), np.linspace(40, 50, n))
        ds = xr.Dataset(
            {"longitude": (["line", "sample"], lon2d),
             "latitude":  (["line", "sample"], lat2d)},
            coords={"line": lines, "sample": samples},
        )
        t0 = time.perf_counter()
        find_pixel(ds, 5.0, 45.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"find_pixel took {elapsed:.2f}s on a 1000×1000 grid"


# ---------------------------------------------------------------------------
# crosses_antimeridian / normalize_longitudes_to_360
# ---------------------------------------------------------------------------

class TestAntimeridian:
    def _make_polygon(self, coords):
        return Polygon(coords)

    def test_normal_polygon_does_not_cross(self):
        poly = self._make_polygon([(-10, 40), (10, 40), (10, 50), (-10, 50), (-10, 40)])
        assert crosses_antimeridian(poly) is False

    def test_antimeridian_crossing_detected(self):
        # Polygon spanning from +170 to -170 (crosses antimeridian)
        poly = self._make_polygon([(170, 40), (-170, 40), (-170, 50), (170, 50), (170, 40)])
        assert crosses_antimeridian(poly) is True

    def test_normalize_negative_longitudes(self):
        poly = self._make_polygon([(-170, 40), (-160, 40), (-160, 50), (-170, 50), (-170, 40)])
        result = normalize_longitudes_to_360(poly)
        assert result is not None
        all_lons = [coord[0] for coord in result.exterior.coords]
        assert all(lon >= 0 for lon in all_lons)

    def test_normalize_already_positive(self):
        poly = self._make_polygon([(10, 40), (20, 40), (20, 50), (10, 50), (10, 40)])
        result = normalize_longitudes_to_360(poly)
        original_lons = [c[0] for c in poly.exterior.coords]
        result_lons   = [c[0] for c in result.exterior.coords]
        assert original_lons == result_lons


# ---------------------------------------------------------------------------
# tools : process_single_tile / add_tiles_footprint
# ---------------------------------------------------------------------------

class TestTools:
    def test_process_single_tile_adds_footprint(self):
        tile = make_tile(n=20)
        result = process_single_tile(tile)
        assert "tile_footprint" in result
        assert "lon_centroid" in result
        assert "lat_centroid" in result

    def test_process_single_tile_centroid_in_bounds(self):
        tile = make_tile(n=20)
        result = process_single_tile(tile)
        lon_min = float(tile["longitude"].min())
        lon_max = float(tile["longitude"].max())
        lat_min = float(tile["latitude"].min())
        lat_max = float(tile["latitude"].max())
        assert lon_min <= float(result.lon_centroid) <= lon_max
        assert lat_min <= float(result.lat_centroid) <= lat_max

    def test_add_tiles_footprint_non_list_raises(self):
        with pytest.raises(ValueError, match="tiles must be a list"):
            add_tiles_footprint("not_a_list")

    def test_add_tiles_footprint_returns_same_count(self):
        tiles = [make_tile(n=20) for _ in range(5)]
        result = add_tiles_footprint(tiles)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Integration test — fully synthetic dataset, no external file needed
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_sar_dataset():
    """
    Minimal xr.Dataset that mimics the structure expected by tiling_prod
    (attrs + variables). Uses detrend=False so no GMF/xsarsea calls are made.
    """
    rng = np.random.default_rng(42)
    n_lines, n_samples, pols = 120, 100, ["VV", "VH"]

    lines   = np.arange(n_lines,   dtype=float)
    samples = np.arange(n_samples, dtype=float)

    shape2 = (n_lines, n_samples)
    shape3 = (len(pols), n_lines, n_samples)

    ds = xr.Dataset(
        {
            "sigma0":         (["pol", "line", "sample"], rng.uniform(0, 1, shape3)),
            "digital_number": (["pol", "line", "sample"], rng.uniform(0, 1, shape3)),
            "land_mask":      (["line", "sample"],        np.zeros(shape2, dtype=bool)),
            "incidence":      (["pol", "line", "sample"], rng.uniform(20, 45, shape3)),
            "nesz":           (["pol", "line", "sample"], rng.uniform(0, 0.1, shape3)),
            "ground_heading": (["line", "sample"],        rng.uniform(0, 360, shape2)),
            "longitude":      (["line", "sample"],        rng.uniform(-5, 5, shape2)),
            "latitude":       (["line", "sample"],        rng.uniform(40, 50, shape2)),
        },
        coords={"line": lines, "sample": samples, "pol": pols},
        attrs={
            "pols":      "VV VH",
            "product":   "GRDH",
            "footprint": "POLYGON ((-5 40, 5 40, 5 50, -5 50, -5 40))",
            "safe":      "S1A_IW_GRDH_1SDV_20210101T000000_20210101T000025_000000_000000.SAFE",
            "swath":     "IW",
            "start_date": "2021-01-01 00:00:00.000000",
            "stop_date":  "2021-01-01 00:00:25.000000",
        },
    )
    return ds


def test_tiling_prod_structure(synthetic_sar_dataset):
    """Validate tiling_prod output structure using a fully synthetic dataset."""
    tile_size = 20   # pixels (resolution=None → 1 m/pixel → nperseg=20)

    _, tiles = grdtiler.tiling_prod(
        path=synthetic_sar_dataset,
        tile_size=tile_size,
        resolution=None,
        detrend=False,
        centering=False,
        noverlap=0,
        save=False,
        add_footprint=True,
    )

    # Dimensions
    assert "tile" in tiles.dims
    assert "tile_line" in tiles.dims
    assert "tile_sample" in tiles.dims
    assert "pol" in tiles.dims
    assert tiles.sizes["tile"] > 0
    assert tiles.sizes["tile_line"] == tile_size
    assert tiles.sizes["tile_sample"] == tile_size

    # Expected variables (detrend=False → no sigma0_detrend)
    for var in ("sigma0", "longitude", "latitude",
                "tile_footprint", "lon_centroid", "lat_centroid"):
        assert var in tiles, f"Missing variable: {var}"

    # Polarisations
    assert set(tiles.pol.values) == {"VV", "VH"}

    # No all-NaN sigma0
    assert not np.all(np.isnan(tiles["sigma0"].values))

    # Footprint comment set
    assert tiles["tile_footprint"].attrs.get("comment") == "Footprint of the tile"

