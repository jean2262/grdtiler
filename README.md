# GRDTILER

Python package for processing, normalising, and tiling Synthetic Aperture Radar (SAR) datasets. Supports Sentinel-1 (S1), Radarsat-2 (RS2), and RCM formats.

---

## Features

- **Grid tiling** — sliding-window tiles with configurable size, overlap, and centering
- **Point-based tiling** — tiles centred around geographic coordinates
- **Wave-mode tiling** — single-tile extraction for Sentinel-1 WV products
- **Sigma0 detrending** — GMF-based detrending via `xsarsea` (CMOD5N, S1_V2, …)
- **Multi-mission support** — S1, RS2, RCM with per-mission GMF configuration
- **Antimeridian handling** — automatic longitude normalisation for crossing products
- **Footprint computation** — WKT polygon + centroid added to every tile
- **NetCDF output** — hierarchical directory structure keyed by mode / size / resolution / date

---

## Installation

### Conda (recommended)

```bash
conda install grdtiler -c oceanscope
```

### From source

```bash
git clone https://github.com/jean2262/grdtiler.git
cd grdtiler
pip install -e .          # runtime dependencies only
pip install -e ".[dev]"   # + pytest / pytest-cov
```

---

## Usage

### 1. Grid tiling — `tiling_prod`

Slides a window across the full dataset and returns all complete tiles.

```python
from grdtiler import tiling_prod

dataset, tiles = tiling_prod(
    path="/path/to/S1A_IW_GRDH.SAFE",
    tile_size=17600,          # metres; or dict {"line": 17600, "sample": 17600}
    resolution="400m",        # str (e.g. "400m") or dict {"line": 20, "sample": 20} in pixels
    detrend=True,
    noverlap=0,               # pixels; or dict {"line": 0, "sample": 0}
    centering=True,
    side="left",
    save=True,
    save_dir="./tiles",
    config_file="grdtiler/config.yaml",
)
```

**Returns** `(dataset, tiles)` — both are `xr.Dataset`.

---

### 2. Point-based tiling — `tiling_by_point`

Extracts one tile centred on each supplied geographic point. Products that cross
the antimeridian are handled automatically — point containment is tested in
normalised [0, 360] longitude space when needed.

```python
from shapely.geometry import Point
from grdtiler import tiling_by_point

locations = [Point(-3.704, 40.417), Point(2.352, 48.857)]

dataset, tiles = tiling_by_point(
    path="/path/to/S1A_IW_GRDH.SAFE",
    posting_loc=locations,
    tile_size=20000,          # metres
    resolution="10m",
    detrend=True,
    save=True,
    save_dir="./point_tiles",
    config_file="grdtiler/config.yaml",
)
```

Points outside the product footprint are skipped with a warning.

---

### 3. Wave-mode tiling — `tiling_wv`

Extracts a single centred tile from a Sentinel-1 WV product. Antimeridian
crossing is handled automatically.

```python
from grdtiler import tiling_wv

dataset, tile = tiling_wv(
    path="/path/to/S1A_WV_SLC.SAFE",
    tile_size=20000,
    resolution="10m",
    detrend=True,
    config_file="grdtiler/config.yaml",
)
```

---

### 4. Standalone detrending — `make_detrend`

Applies GMF-based sigma0 detrending to any pre-loaded SAR dataset without
triggering the full tiling pipeline.

```python
from grdtiler import make_detrend

dataset = make_detrend(
    path="/path/to/S1A_IW_GRDH.SAFE",   # or xr.Dataset
    config="grdtiler/config.yaml",        # path or pre-loaded dict
    resolution="400m",
)
# dataset now contains sigma0_no_nan and sigma0_detrend variables
```

---

### 5. Antimeridian utilities — `crosses_antimeridian`, `normalize_longitudes_to_360`

```python
from grdtiler import crosses_antimeridian, normalize_longitudes_to_360
from shapely import wkt

footprint = wkt.loads(dataset.attrs["footprint"])

if crosses_antimeridian(footprint):
    footprint_360 = normalize_longitudes_to_360(footprint)
```

---

## API reference

### `tiling_prod`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| xr.Dataset` | — | Path to SAR product or pre-loaded dataset |
| `tile_size` | `int \| dict[str, int]` | — | Tile size in metres (`{"line": …, "sample": …}` for non-square) |
| `resolution` | `str \| dict \| None` | `None` | Target resolution, e.g. `"400m"` (string) or `{"line": 20, "sample": 20}` (pixels per axis) |
| `detrend` | `bool` | `True` | Apply GMF-based sigma0 detrending |
| `noverlap` | `int \| dict[str, int]` | `0` | Overlap in pixels between adjacent tiles |
| `centering` | `bool` | `False` | Centre the tiling grid inside the dataset |
| `side` | `str` | `"left"` | Alignment when centering (`"left"` or `"right"`) |
| `add_footprint` | `bool` | `True` | Compute and attach tile footprint + centroid |
| `save` | `bool` | `False` | Save tiles to NetCDF |
| `save_dir` | `str` | `"."` | Root directory for saved tiles |
| `to_keep_var` | `list[str] \| None` | `None` | Variables to retain (default set if `None`) |
| `config_file` | `str` | `"config.yaml"` | Path to GMF model configuration |

### `tiling_by_point`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| xr.Dataset` | — | Path to SAR product or pre-loaded dataset |
| `posting_loc` | `list[Point]` | — | Shapely `Point` objects (lon, lat) |
| `tile_size` | `int` | — | Tile size in metres |
| `resolution` | `str \| dict \| None` | `None` | Target resolution, e.g. `"400m"` or `{"line": 20, "sample": 20}` (pixels per axis) |
| `detrend` | `bool` | `True` | Apply sigma0 detrending |
| `scat_info` | `dict \| None` | `None` | Scatterometer wind data to annotate tiles |
| `save` | `bool` | `False` | Save tiles to NetCDF |
| `save_dir` | `str` | `"."` | Root directory for saved tiles |
| `to_keep_var` | `list[str] \| None` | `None` | Variables to retain |
| `config_file` | `str` | `"config.yaml"` | Path to GMF model configuration |

### `tiling_wv`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Path to Sentinel-1 WV SAFE product |
| `tile_size` | `int` | — | Tile size in metres |
| `resolution` | `str \| dict \| None` | `None` | Target resolution, e.g. `"400m"` or `{"line": 20, "sample": 20}` (pixels per axis) |
| `detrend` | `bool` | `True` | Apply sigma0 detrending |
| `to_keep_var` | `list[str] \| None` | `None` | Variables to retain |
| `config_file` | `str` | `"config.yaml"` | Path to GMF model configuration |

### `make_detrend`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| xr.Dataset` | — | Path to SAR product or pre-loaded dataset |
| `config` | `str \| dict` | — | Path to GMF config YAML or pre-loaded dict |
| `resolution` | `str \| dict \| None` | `None` | Target resolution, e.g. `"400m"` or `{"line": 20, "sample": 20}` (pixels per axis) |

### `crosses_antimeridian`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `polygon` | `Polygon \| MultiPolygon` | — | Shapely geometry to test |
| `threshold` | `float` | `150` | Longitude threshold for crossing detection |

### `normalize_longitudes_to_360`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `polygon` | `Polygon` | — | Shapely polygon with coordinates in [-180, 180] |

---

## GMF configuration

The `config.yaml` file maps each mission to its GMF model names:

```yaml
gmf_base_path: "/path/to/gmf/nc_luts"

gmf_models:
    S1:
        GMF_VV_NAME: "gmf_cmod5n"
        GMF_VH_NAME: "gmf_s1_v2"
        GMF_HH_NAME: "nc_lut_gmf_cmod7_Rhigh_hh_zhangA"
        GMF_HV_NAME: "nc_lut_gmf_cmod7_Rhigh_hh_zhangA"
    RS2:
        GMF_VV_NAME: "gmf_cmod5n"
        GMF_VH_NAME: "gmf_rs2_v2"
    RCM:
        GMF_VV_NAME: "gmf_cmod5n"
        GMF_VH_NAME: "gmf_rcm_noaa"
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 29 tests are self-contained — no SAR product download required.

---

## License

MIT — see [`LICENSE`](LICENSE).
