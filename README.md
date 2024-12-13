# GRDTILER

This repository contains Python functions to process, normalize, and tile Synthetic Aperture Radar (SAR) datasets. The toolkit provides utilities to create tiles, normalize SAR data, and manage detrending processes, while supporting several SAR dataset formats.

---

## Features
- **Dataset Tiling:** Generate spatially consistent tiles from SAR datasets with configurable overlap and centering options.
- **Detrending:** Apply detrending processes based on dataset polarization.
- **Multi-Format Support:** Handle SAR dataset types such as GRD, RS2, and RCM.
- **Point-based Tiling:** Generate tiles around specific geographic locations.
- **Customizable Variables:** Retain only necessary variables in the processed tiles.
- **Tile Saving:** Save processed tiles in NetCDF format.

---

## Installation

Install the grdtiler using pip:

```bash
pip install grdtiler
```

Or clone the repository and install dependencies manually:

```bash
git clone https://github.com/jean2262/grdtiler.git
cd grdtiler
```

---

## Requirements

To use this toolkit, install the following dependencies:

- `numpy`
- `xarray`
- `xsar`
- `xsarsea`
- `shapely`
- `tqdm`
- `logging`

Install these dependencies using:

```bash
pip install numpy xarray xsar xsarsea shapely tqdm
```

---

## Usage

### 1. Tiling SAR Datasets

```python
from grdtiler import tiling_prod

dataset, tiles = tiling_prod(
    path="/path/to/sar/dataset",
    tile_size=17600,
    resolution="400m",
    detrend=True,
    noverlap=32,
    centering=False,
    save=True,
    save_dir="./tiles",
)
```

### 2. Detrending SAR Datasets

```python
from grdtiler import make_detrend

processed_path = make_detrend(
    path="/path/to/sar/dataset",
    resolution="10m",
    save_dir="./processed",
)
```

### 3. Point-Based Tiling

```python
from shapely.geometry import Point
from grdtiler import tiling_by_point

posting_locations = [Point(-3.70379, 40.41678), Point(2.35222, 48.85661)]
dataset, point_tiles = tiling_by_point(
    path="/path/to/sar/dataset",
    posting_loc=posting_locations,
    tile_size=1024,
    resolution="10m",
    detrend=True,
    save=True,
    save_dir="./point_tiles",
)
```

---

## Key Functions

### `tiling_prod`
Tiles SAR datasets into square tiles.

**Parameters:**
- `path` (str): Path to the dataset.
- `tile_size` (int or dict): Size of the tiles in meters.
- `resolution` (str, optional): Resolution of the dataset.
- `save` (bool, optional): Whether to save the tiles.

**Returns:**
- Processed dataset and tiles.

### `make_detrend`
Applies detrending to SAR datasets.

**Parameters:**
- `path` (str): Path to the dataset.
- `resolution` (str): Dataset resolution.
- `save_dir` (str, optional): Directory to save the processed dataset.

**Returns:**
- Processed dataset.

### `tiling_by_point`
Generates tiles centered around specified geographic points.

**Parameters:**
- `path` (str): Path to the dataset.
- `posting_loc` (list): List of `shapely.geometry.Point` objects.
- `tile_size` (int): Size of tiles in meters.

**Returns:**
- Processed dataset and point-centered tiles.

---

## Logging

This toolkit uses Python's logging module for detailed progress information. To enable logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
