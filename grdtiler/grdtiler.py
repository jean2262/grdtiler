import logging
import yaml

import numpy as np
import xarray as xr
import xsar
import xsarsea
from shapely import Point, Polygon, MultiPolygon
from shapely import wkt
from tqdm import tqdm
from pathlib import Path

from grdtiler.tools import add_tiles_footprint, save_tile

logger = logging.getLogger(__name__)

PIXELSPACING = {
    "S1_IW_GRDH": 10,
    "S1_IW_GRDM": 40,
    "S1_EW_GRDH": 40,
    "S1_EW_GRDV": 40,
}


def _detect_mission(filename: str) -> str:
    """Return the SAR mission code ('S1', 'RS2', 'RCM') from a product filename."""
    name = filename.upper()
    if name.startswith("S1"):
        return "S1"
    elif name.startswith("RS2"):
        return "RS2"
    elif name.startswith("RCM"):
        return "RCM"
    raise ValueError(f"Unsupported mission in filename: {filename}. Expected S1, RS2 or RCM.")

def tiling_prod(
    path: str|xr.Dataset,
    tile_size: int,
    resolution: str|int=None,
    detrend: bool=True,
    noverlap: int=0,
    centering: bool=False,
    side: str="left",
    save: bool=False,
    save_dir: str=".",
    to_keep_var: list=None,
    add_footprint: bool=True,
    config_file: str="config.yaml",
) -> xr.Dataset:

    """
    Slide a window across a SAR dataset and return all complete tiles.

    Args:
        path (str or xr.Dataset): Path to the SAR product (.SAFE or .nc) or a
            pre-loaded xarray Dataset.
        tile_size (int or dict): Tile size in metres. Use an int for square tiles
            or ``{"line": H, "sample": W}`` for rectangular ones.
        resolution (str, optional): Target resolution string (e.g. ``"400m"``).
            Passed to ``xsar.open_dataset``. Defaults to None.
        detrend (bool, optional): Apply GMF-based sigma0 detrending. Defaults to True.
        noverlap (int or dict, optional): Overlap in pixels between adjacent tiles.
            Use an int or ``{"line": n, "sample": m}``. Defaults to 0.
        centering (bool, optional): Centre the tiling grid inside the dataset so
            that incomplete border tiles are discarded symmetrically. Defaults to False.
        side (str, optional): Alignment bias when centering (``"left"`` or ``"right"``).
            Defaults to ``"left"``.
        save (bool, optional): Write tiles to NetCDF. Defaults to False.
        save_dir (str, optional): Root directory for saved tiles. Defaults to ``"."``.
        to_keep_var (list, optional): Data variables to retain. When None a default
            set is used (sigma0, land_mask, incidence, nesz, …). Defaults to None.
        add_footprint (bool, optional): Compute and attach the WKT footprint polygon
            and lon/lat centroid for each tile. Defaults to True.
        config_file (str, optional): Path to the GMF model configuration YAML.
            Defaults to ``"config.yaml"``.

    Returns:
        tuple[xr.Dataset, xr.Dataset]: The normalised full dataset and the
        concatenated tile dataset (dim ``"tile"``).

    Raises:
        ValueError: If the path points to an unsupported file or dataset type.
        ValueError: If no tiles can be generated (dataset smaller than tile_size).
    """

    dataset = load_dataset(path, resolution=resolution)
    logger.info("Start tiling...")

    dataset, nperseg = tile_normalize(
        dataset=dataset, tile_size=tile_size, resolution=resolution, noverlap=noverlap, detrend=detrend, to_keep_var=to_keep_var, config_file=config_file
    )
    tiles = tiling(
        dataset=dataset,
        tile_size=nperseg,
        noverlap=noverlap,
        centering=centering,
        side=side,
        add_footprint=add_footprint,
    )

    logger.info("Done tiling...")

    if save:
        save_tile(tiles, save_dir)

    return dataset, tiles

def load_dataset(path, resolution=None):
    """
    Load a SAR dataset from a file path or return a pre-loaded Dataset unchanged.

    Args:
        path (str or xr.Dataset): Path to a ``.nc`` file, a SAR SAFE product
            (prefix S1 / RS2 / RCM), or an already-loaded ``xr.Dataset``.
        resolution (str, optional): Target resolution forwarded to
            ``xsar.open_dataset`` (e.g. ``"400m"``). Ignored for NetCDF files
            and pre-loaded datasets. Defaults to None.

    Returns:
        xr.Dataset: The loaded dataset.

    Raises:
        ValueError: If the file extension or product prefix is not supported.
    """
    if isinstance(path, xr.Dataset):
        dataset = path
    else:
        path_obj = Path(path)
        
        if path_obj.suffix.lower() == ".nc":
            dataset = xr.open_dataset(path_obj)
        else:
            SUPPORTED_PREFIXES = ("S1", "RS2", "RCM")
            if path_obj.name.startswith(SUPPORTED_PREFIXES):
                dataset = xsar.open_dataset(str(path_obj), resolution=resolution)
            else:
                raise ValueError(
                    f"Unsupported file or dataset type: {path}. "
                    f"Expected a .nc file or a SAR product starting with {SUPPORTED_PREFIXES}."
                )

    return dataset

def load_gmf_model(filename, pol, config_file=None, config=None):
    """
    Return the GMF model name for the given filename and polarisation.

    Parameters
    ----------
    filename : str
        SAR product filename (used to detect the mission).
    pol : str
        Polarisation, one of 'HH', 'HV', 'VV', 'VH'.
    config_file : str, optional
        Path to the YAML configuration file. Required when *config* is None.
    config : dict, optional
        Pre-loaded configuration dict. When provided, *config_file* is ignored,
        avoiding redundant file I/O inside loops.

    Returns
    -------
    str
        Name of the GMF model to use.
    """
    if config is None:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

    mission = _detect_mission(filename)

    if pol in ("HH", "HV"):
        xsarsea.windspeed.register_nc_luts(config["gmf_base_path"])

    return config['gmf_models'][mission][f'GMF_{pol}_NAME']

def move_valid_line_to_zero(inc_angle):
    """
    Shift the line coordinate so that the first fully-valid line becomes index 0.

    SAR datasets may have leading NaN rows (e.g. from burst boundaries). This
    function slices away those rows and reassigns the ``line`` coordinate to a
    zero-based integer sequence, which is required by ``xsarsea.sigma0_detrend``.

    Args:
        inc_angle (xr.DataArray): Incidence angle array with a ``line`` dimension.

    Returns:
        xr.DataArray: Sliced array with ``line`` coordinate reset to ``0, 1, 2, …``.
    """
    nan_mask = np.isnan(inc_angle.values)
    nan_counts = nan_mask.sum(axis=tuple(range(1, nan_mask.ndim)))
    valid_lines = np.where(nan_counts == 0)[0]
    if valid_lines.size > 0:
        first_valid_line = valid_lines[0]
    else:
        first_valid_line = np.argmin(nan_counts) 

    sliced = inc_angle.isel(line=slice(first_valid_line, None))
    return sliced.assign_coords(line=np.arange(sliced.sizes['line']))

def tile_normalize(dataset, tile_size, resolution, noverlap=0, detrend=True, to_keep_var=None, config_file="config.yaml"):
    """
    Prepare a SAR dataset for tiling: set metadata attrs, compute sigma0 detrend,
    and drop unrequested variables.

    Args:
        dataset (xr.Dataset): Input SAR dataset (must expose ``pols``, ``product``,
            and ``footprint`` attributes).
        tile_size (int or dict): Tile size in metres. Use an int for square tiles
            or ``{"line": H, "sample": W}`` for rectangular ones.
        resolution (str): Resolution string used to convert metres to pixel count
            (e.g. ``"400m"`` → divide by 400). Pass ``None`` to assume 1 m/pixel.
        noverlap (int or dict, optional): Overlap in pixels; forwarded to attrs only.
            Defaults to 0.
        detrend (bool, optional): Compute and attach ``sigma0_detrend`` using the
            appropriate GMF model from *config_file*. Defaults to True.
        to_keep_var (list, optional): Variables to retain. When None the default set
            ``["digital_number", "sigma0", "land_mask", "ground_heading",
            "incidence", "nesz", "longitude", "latitude"]`` is used. Defaults to None.
        config_file (str, optional): Path to the GMF model configuration YAML.
            Defaults to ``"config.yaml"``.

    Returns:
        tuple[xr.Dataset, int | dict]: The normalised dataset and the tile size in
        pixels (int, or ``{"line": …, "sample": …}``).
    """
    default_vars = ["longitude", "latitude"]
    
    if to_keep_var is not None:
        to_keep_var.extend(default_vars)
    else:
        to_keep_var = ["digital_number", "sigma0", "land_mask", "ground_heading", "incidence", "nesz"]
        to_keep_var.extend(default_vars)

    resolution_value = int(resolution.split("m")[0]) if resolution else 1

    if isinstance(tile_size, dict):
        tile_line_size = tile_size.get("line", 1)
        tile_sample_size = tile_size.get("sample", 1)
        nperseg = {
            "line": tile_line_size // resolution_value,
            "sample": tile_sample_size // resolution_value,
        }
        dataset.attrs["tile_size"] = (
            f"{tile_line_size}m*{tile_sample_size}m (line * sample)"
        )
    else:
        nperseg = tile_size // resolution_value
        dataset.attrs["tile_size"] = f"{tile_size}m*{tile_size}m (line * sample)"

    dataset.attrs.update(
        {
            "resolution": resolution,
            "noverlap": f"{noverlap} pixels",
            "polarizations": dataset.attrs["pols"],
            "processing_level": dataset.attrs["product"],
            "main_footprint": dataset.attrs["footprint"],
        }
    )

    if detrend:
        dataset["sigma0_no_nan"] = xr.where(
            dataset["land_mask"], np.nan, dataset["sigma0"]
        )
        filename = dataset.attrs["safe"] if "safe" in dataset.attrs else dataset.attrs["name"]
        if ":" in filename:
            filename = Path(filename.split(":")[1]).name

        # Load config and detect mission once — avoids repeated file I/O inside the loop
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        mission = _detect_mission(filename)

        # Register nc LUTs once if any cross-pol is present
        pols = list(dataset.pol.values)
        if any(p in ("HH", "HV") for p in pols):
            xsarsea.windspeed.register_nc_luts(config["gmf_base_path"])

        sigma0_detrends = []
        for pol in pols:
            gmf_model = config['gmf_models'][mission][f'GMF_{pol}_NAME']
            inc_angle_cleaned = move_valid_line_to_zero(dataset.sel(pol=pol).incidence)
            sigma0_detrends.append(xsarsea.sigma0_detrend(
                sigma0=dataset.sel(pol=pol).sigma0,
                inc_angle=inc_angle_cleaned,
                model=gmf_model,
            ))
        dataset["sigma0_detrend"] = xr.concat(sigma0_detrends, dim="pol")

        to_keep_var.append("sigma0_detrend")

    if "longitude" in dataset.variables and "latitude" in dataset.variables:
        dataset["sigma0"] = dataset["sigma0"].transpose(*dataset["sigma0"].dims)

    dataset = dataset.drop_vars(set(dataset.data_vars) - set(to_keep_var))
    
    if "product_path" in dataset.attrs:
        dataset.attrs["safe"] = Path(dataset.attrs["product_path"]).name
        
    attributes_to_remove = {
        "multidataset",
        "product",
        "pols",
        "footprint",
        "platform_heading",
        "short_name",
        "product_path",
        "rawDataStartTime",
        "approx_transform"
    }
        
    dataset.attrs = {
        key: value
        for key, value in dataset.attrs.items()
        if key not in attributes_to_remove
    }

    return dataset, nperseg


def tiling(dataset, tile_size, noverlap, centering, side, add_footprint=True):
    """
    Extract all complete tiles from a SAR dataset using a sliding window.

    Dimensions ``line`` and ``sample`` are renamed to ``tile_line`` and
    ``tile_sample`` in the output; all tiles are concatenated along a new
    ``tile`` dimension.

    Args:
        dataset (xr.Dataset): Normalised SAR dataset with ``line`` and ``sample``
            dimension coordinates.
        tile_size (int or dict): Tile size in **pixels**. Use an int for square
            tiles or ``{"line": H, "sample": W}`` for rectangular ones.
        noverlap (int or dict): Overlap in pixels between adjacent tiles. Use an
            int or ``{"line": n, "sample": m}``.
        centering (bool): When True, compute the largest centred sub-region that
            fits complete tiles and discard the border remainder symmetrically.
        side (str): Bias direction for the centering offset (``"left"`` or
            ``"right"``).
        add_footprint (bool, optional): Compute and attach ``tile_footprint``
            (WKT), ``lon_centroid``, and ``lat_centroid`` for each tile.
            Defaults to True.

    Returns:
        xr.Dataset: Concatenated tile dataset with dimensions
        ``(tile, pol, tile_line, tile_sample)``.

    Raises:
        ValueError: If overlap ≥ tile size in either dimension.
        ValueError: If no complete tiles fit within the dataset (or centred mask).
    """
    tiles = []
    tile_line_size, tile_sample_size = (
        (tile_size.get("line", 1), tile_size.get("sample", 1))
        if isinstance(tile_size, dict)
        else (tile_size, tile_size)
    )
    line_overlap, sample_overlap = (
        (noverlap.get("line", 0), noverlap.get("sample", 0))
        if isinstance(noverlap, dict)
        else (noverlap, noverlap)
    )

    total_lines, total_samples = dataset.sizes["line"], dataset.sizes["sample"]
    mask = dataset

    if line_overlap >= tile_line_size or sample_overlap >= tile_sample_size:
        raise ValueError("Overlap size must be less than tile size")

    if centering:
        complete_segments_line = (total_lines - tile_line_size) // (
            tile_line_size - line_overlap
        ) + 1
        mask_size_line = (
            complete_segments_line * tile_line_size
            - (complete_segments_line - 1) * line_overlap
        )

        complete_segments_sample = (total_samples - tile_sample_size) // (
            tile_sample_size - sample_overlap
        ) + 1
        mask_size_sample = (
            complete_segments_sample * tile_sample_size
            - (complete_segments_sample - 1) * sample_overlap
        )

        if side == "right":
            start_line = (total_lines // 2) - (mask_size_line // 2)
            start_sample = (total_samples // 2) - (mask_size_sample // 2)
        else:
            start_line = (total_lines // 2) + (total_lines % 2) - (mask_size_line // 2)
            start_sample = (
                (total_samples // 2) + (total_samples % 2) - (mask_size_sample // 2)
            )

        mask = dataset.isel(
            line=slice(start_line, start_line + mask_size_line),
            sample=slice(start_sample, start_sample + mask_size_sample),
        )

    step_line = tile_line_size - line_overlap
    step_sample = tile_sample_size - sample_overlap
    mask_lines = mask.sizes["line"]
    mask_samples = mask.sizes["sample"]

    for line_start in tqdm(
        range(0, mask_lines - tile_line_size + 1, step_line), desc="Tiling"
    ):
        for sample_start in range(0, mask_samples - tile_sample_size + 1, step_sample):
            subset = mask.isel(
                line=slice(line_start, line_start + tile_line_size),
                sample=slice(sample_start, sample_start + tile_sample_size),
            )
            tiles.append(
                subset.drop_indexes(["line", "sample"]).rename_dims(
                    {"line": "tile_line", "sample": "tile_sample"}
                )
            )
    if not tiles:
        raise ValueError("No tiles generated")
    
    if add_footprint:
        tiles_with_footprint = add_tiles_footprint(tiles)
    else:
        tiles_with_footprint = tiles
        
    all_tiles = xr.concat(tiles_with_footprint, dim="tile")
    
    if add_footprint:
        all_tiles["tile_footprint"].attrs["comment"] = "Footprint of the tile"
        all_tiles["lon_centroid"].attrs["comment"] = (
            "Longitude of the tile footprint's centroid"
        )
        all_tiles["lat_centroid"].attrs["comment"] = (
            "Latitude of the tile footprint's centroid"
        )

    return all_tiles

def find_pixel(ds, lon0, lat0):
    """
    Return the (line, sample) coordinates of the pixel closest to (lon0, lat0).

    Uses a coarse-to-fine two-pass search:
      1. Coarse pass on a subsampled grid to locate the approximate region.
      2. Fine pass over a small window around that region at native resolution.

    For a 25 000 × 16 000 image with step=50 this reduces the number of
    distance evaluations from ~400 M to ~160 000 + ~10 000.
    """
    lon = ds.longitude.compute().values
    lat = ds.latitude.compute().values

    # --- Coarse pass ---
    step = max(1, min(lon.shape) // 50)
    lon_c = lon[::step, ::step]
    lat_c = lat[::step, ::step]
    iy_c, ix_c = np.unravel_index(
        np.argmin((lon_c - lon0) ** 2 + (lat_c - lat0) ** 2), lon_c.shape
    )

    # --- Fine pass in a window around the coarse result ---
    margin = step + 1
    r0 = max(0, iy_c * step - margin)
    r1 = min(lon.shape[0], iy_c * step + margin + 1)
    c0 = max(0, ix_c * step - margin)
    c1 = min(lon.shape[1], ix_c * step + margin + 1)

    lon_f = lon[r0:r1, c0:c1]
    lat_f = lat[r0:r1, c0:c1]
    iy_f, ix_f = np.unravel_index(
        np.argmin((lon_f - lon0) ** 2 + (lat_f - lat0) ** 2), lon_f.shape
    )

    return (ds.line.values[r0 + iy_f], ds.sample.values[c0 + ix_f])

def tiling_by_point(
    path,
    posting_loc,
    tile_size,
    resolution=None,
    detrend=True,
    save=False,
    save_dir=".",
    to_keep_var=None,
    scat_info=None,
    config_file="config.yaml",
):
    """
    Extract one tile centred on each supplied geographic point.

    Points that fall outside the product footprint or that cannot be mapped to
    valid pixel coordinates are skipped with a warning. If all points are
    skipped the function returns ``(dataset, None)``.

    Args:
        path (str or xr.Dataset): Path to the SAR product or a pre-loaded dataset.
        posting_loc (list[Point]): Shapely ``Point`` objects in (lon, lat) order.
        tile_size (int): Tile side length in metres.
        resolution (str, optional): Target resolution (e.g. ``"10m"``). Defaults to None.
        detrend (bool, optional): Apply GMF-based sigma0 detrending. Defaults to True.
        save (bool, optional): Write tiles to NetCDF. Defaults to False.
        save_dir (str, optional): Root directory for saved tiles. Defaults to ``"."``.
        to_keep_var (list, optional): Variables to retain. Defaults to None.
        scat_info (dict, optional): Scatterometer wind data to annotate tiles.
            Expected keys: ``"wind_direction"`` and ``"wind_speed"`` (indexed by
            point position). Defaults to None.
        config_file (str, optional): Path to the GMF configuration YAML.
            Defaults to ``"config.yaml"``.

    Returns:
        tuple[xr.Dataset, xr.Dataset | None]: The normalised full dataset and the
        concatenated tile dataset, or ``None`` if no valid tiles were produced.

    Raises:
        ValueError: If any entry in *posting_loc* is ``None``.
    """
    dataset = load_dataset(path, resolution=resolution)
    
    logger.info("Start tiling...")
    
    tiles = []
    filename = dataset.attrs["safe"] if "safe" in dataset.attrs else dataset.attrs["name"]
    if ":" in filename:
        filename = Path(filename.split(":")[1]).name
    
    parts = filename.replace("-", "_").split("_")
    fn = "_".join(parts[:3])
    
    footprint = dataset.attrs["footprint"]
    footprint = wkt.loads(footprint) if isinstance(footprint, str) else footprint
    dataset, nperseg = tile_normalize(
        dataset=dataset, tile_size=tile_size, resolution=resolution, detrend=detrend, to_keep_var=to_keep_var, config_file=config_file
    )
    for i, point in tqdm(enumerate(posting_loc), total=len(posting_loc), desc="Tiling"):
        if point is None:
            raise ValueError(f"Invalid posting location: {posting_loc}")

        if not footprint.contains(point):
            logger.warning(f"Skipping {point} as it is outside the footprint.")
            continue

        point_coords = find_pixel(dataset, point.x, point.y)

        if np.isnan(point_coords).any():
            logger.warning(
                f"Choose a point inside the footprint: {footprint}, for: {point}"
            )
            continue

        if filename.upper().startswith("S1"):
            pixel_spacing = PIXELSPACING[fn.upper()]
        else:
            pixel_spacing = dataset.pixel_line_m
            
        dist = {
            "line": int(np.round(tile_size / 2 / pixel_spacing)),
            "sample": int(np.round(tile_size / 2 / pixel_spacing)),
        }
        
        line_start = point_coords[0] - dist["line"]
        line_end = point_coords[0] + dist["line"]
        sample_start = point_coords[1] - dist["sample"]
        sample_end = point_coords[1] + dist["sample"]
        
        tile = dataset.sel(
            line=slice(line_start, line_end),
            sample=slice(sample_start, sample_end),
        )
        
        expected_line = nperseg if isinstance(nperseg, int) else nperseg["line"]
        expected_sample = nperseg if isinstance(nperseg, int) else nperseg["sample"]
        
        # Get coordinate spacing (assuming it's uniform)
        line_coords = dataset["line"].values
        sample_coords = dataset["sample"].values
        line_step = abs(line_coords[1] - line_coords[0])
        sample_step = abs(sample_coords[1] - sample_coords[0])
        
        # Adjust for off-by-one differences
        actual_line = tile.sizes["line"]
        actual_sample = tile.sizes["sample"]
        
        if abs(actual_line - expected_line) == 1:
            if actual_line < expected_line:
                line_end += line_step  # extend by one pixel equivalent
            else:
                line_end -= line_step  # reduce by one pixel equivalent
        
        if abs(actual_sample - expected_sample) == 1:
            if actual_sample < expected_sample:
                sample_end += sample_step
            else:
                sample_end -= sample_step
        
        # Re-select with corrected bounds
        tile = dataset.sel(
            line=slice(line_start, line_end),
            sample=slice(sample_start, sample_end),
        )

        if isinstance(nperseg, int):
            if not tile.sizes["line"] == nperseg or not tile.sizes["sample"] == nperseg:
                logger.warning(f"Error on tile size {tile.sizes}, for {point}")
                continue
        else:
            if (
                not tile.sizes["line"] == nperseg["line"]
                or not tile.sizes["sample"] == nperseg["sample"]
            ):
                logger.warning(f"Error on tile size {tile.sizes}, for {point}")
                continue
            
        tile = tile.assign(origin_point=str(point))
        if scat_info:
            tile = tile.assign(
                scat_wind_direction=scat_info["wind_direction"][i],
                scat_wind_speed=scat_info["wind_speed"][i],
            )
        tiles.append(
            tile.drop_indexes(["line", "sample"]).rename_dims(
                {"line": "tile_line", "sample": "tile_sample"}
            )
        )

    logger.info("Done tiling...")

    if len(tiles) == 0:
        logger.warning("No tiles generated.")
        return dataset, None
    else:
        tiles = add_tiles_footprint(tiles)
        all_tiles = xr.concat(tiles, dim="tile")
        all_tiles["origin_point"].attrs["comment"] = "Origin input points"
        all_tiles["tile_footprint"].attrs["comment"] = "Footprint of the tile"
        all_tiles["lon_centroid"].attrs["comment"] = (
            "Longitude of the tile footprint's centroid"
        )
        all_tiles["lat_centroid"].attrs["comment"] = (
            "Latitude of the tile footprint's centroid"
        )
        if "scat_wind_direction" in list(all_tiles.variables):
            all_tiles["scat_wind_direction"].attrs["comment"] = (
                "Geographic reference (degrees)"
            )
            all_tiles["scat_wind_speed"].attrs["comment"] = (
                "Wind speed in meters per second (m/s)"
            )

        if save:
            save_tile(all_tiles, save_dir)

        return dataset, all_tiles

def crosses_antimeridian(polygon, threshold=150):
    """
    Return True if *polygon* crosses the antimeridian (±180° longitude).

    Detects crossing by inspecting consecutive vertex pairs: if the absolute
    longitude difference exceeds 180° the edge must wrap around the antimeridian.
    ``MultiPolygon`` geometries are first merged into a single polygon.

    Args:
        polygon (Polygon or MultiPolygon): Shapely geometry to test.
        threshold (float, optional): Minimum absolute longitude (degrees) that
            both vertices must exceed before the crossing check is applied.
            Defaults to 150.

    Returns:
        bool: True if the polygon crosses the antimeridian, False otherwise.
    """
    if isinstance(polygon, MultiPolygon):
        polygon = polygon.geoms[0].union(polygon.geoms[1])
    coords = list(polygon.exterior.coords)

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        
        if (lon1 > threshold and lon2 < -threshold) or (lon1 < -threshold and lon2 > threshold):
            # Only return True if there is an actual crossing
            if abs(lon1 - lon2) > 180:
                return True
    return False

def normalize_longitudes_to_360(polygon):
    """
    Convert all negative longitudes in *polygon* to the [0, 360] range.

    Useful when a footprint spans the antimeridian and standard [-180, 180]
    coordinates produce a degenerate geometry. Returns ``None`` on failure so
    callers can handle the error without an exception.

    Args:
        polygon (Polygon): Shapely polygon with coordinates in [-180, 180].

    Returns:
        Polygon or None: New polygon with longitudes in [0, 360], or ``None``
        if the conversion raised an exception.
    """
    try:
        normalized_coords = [
            ((lon + 360) if lon < 0 else lon, lat) 
            for lon, lat in polygon.exterior.coords
        ]
        return Polygon(normalized_coords)
    except Exception as e:
        logger.error(f"Error normalizing longitudes: {e}")
        return None


def tiling_wv(
    path,
    tile_size,
    resolution=None,
    detrend=True,
    to_keep_var=None,
    config_file="config.yaml",
):
    """
    Tile a Sentinel-1 Wave (WV) mode product into a single centred tile.

    Handles antimeridian crossing by normalising longitudes to [0, 360] before
    computing the footprint centroid.

    Args:
        path (str): Path to the Sentinel-1 WV SAFE product.
        tile_size (int): Size of the tile in metres.
        resolution (str, optional): Target resolution (e.g. '10m'). Defaults to None.
        detrend (bool, optional): Whether to apply sigma0 detrending. Defaults to True.
        to_keep_var (list, optional): Variables to keep in the output. Defaults to None.
        config_file (str, optional): Path to the GMF config YAML. Defaults to 'config.yaml'.

    Returns:
        tuple[xr.Dataset, xr.Dataset | None]:
            The full normalised dataset and the tiled dataset (or None on failure).

    Raises:
        ValueError: If *path* does not point to a Sentinel-1 WV product.
    """
    if "WV" in path.split("/")[-1]:
        s1meta = xsar.sentinel1_meta.Sentinel1Meta(path)
        s1ds = xsar.sentinel1_dataset.Sentinel1Dataset(path, resolution=resolution)
        s1ds.add_high_resolution_variables()
        s1ds.apply_calibration_and_denoising()
        dataset = s1ds.dataset
    else:
        raise ValueError("Path must be a Sentinel-1 WV path.")

    dataset, nperseg = tile_normalize(
        dataset=dataset, tile_size=tile_size, resolution=resolution, detrend=detrend, to_keep_var=to_keep_var, config_file=config_file
    )
    # main_footprint is stored as a string attr by tile_normalize — parse it
    raw_footprint = dataset.attrs["main_footprint"]
    main_footprint = wkt.loads(raw_footprint) if isinstance(raw_footprint, str) else raw_footprint

    is_cross_antimeridian = crosses_antimeridian(main_footprint)

    if is_cross_antimeridian:
        correct_footprint = normalize_longitudes_to_360(main_footprint)
        if correct_footprint is None:
            logger.warning("Unable to normalize longitudes to 360 degrees.")
            return dataset, None
        point = correct_footprint.centroid

        line_mid = dataset.sizes["line"] // 2
        sample_mid = dataset.sizes["sample"] // 2

        tile = dataset.isel(
            line=slice(line_mid - (nperseg // 2), line_mid + (nperseg // 2)),
            sample=slice(sample_mid - (nperseg // 2), sample_mid + (nperseg // 2))
        )
    else:
        point = main_footprint.centroid
        lon, lat = point.x, point.y

        if not main_footprint.contains(Point(lon, lat)):
            logger.warning(f"Skipping point ({lon}, {lat}) as it is outside the footprint.")
            return dataset, None

        point_coords = s1ds.ll2coords(lon, lat)

        dist = {
            "line": int(np.round(tile_size / (2 * s1meta.pixel_line_m))),
            "sample": int(np.round(tile_size / (2 * s1meta.pixel_sample_m))),
        }

        tile = dataset.sel(
            line=slice(
                point_coords[0] - dist["line"], point_coords[0] + dist["line"] - 1
            ),
            sample=slice(
                point_coords[1] - dist["sample"], point_coords[1] + dist["sample"] - 1
            ),
        )

        # Adjust start index per axis if off by one pixel
        line_adj   = -1 if tile.sizes["line"]   < nperseg else 0
        sample_adj = -1 if tile.sizes["sample"] < nperseg else 0
        if line_adj or sample_adj:
            tile = dataset.sel(
                line=slice(
                    point_coords[0] - dist["line"]   + line_adj,
                    point_coords[0] + dist["line"]   - 1,
                ),
                sample=slice(
                    point_coords[1] - dist["sample"] + sample_adj,
                    point_coords[1] + dist["sample"] - 1,
                ),
            )

    if isinstance(nperseg, int):
        if tile.sizes["line"] != nperseg or tile.sizes["sample"] != nperseg:
            logger.error(f"Incorect tile size {tile.sizes} for {point}")
            return dataset, None
    else:
        if (
            tile.sizes["line"] != nperseg["line"]
            or tile.sizes["sample"] != nperseg["sample"]
        ):
            logger.error(f"Incorect tile size {tile.sizes} for {point}")
            return dataset, None
    safe_name = path.split("/")[-1]

    tile = tile.assign(
        origin_point=str(point),
        origin_safe=str(safe_name)
        )

    tiles = [
        tile.drop_indexes(["line", "sample"]).rename_dims(
            {"line": "tile_line", "sample": "tile_sample"}
        )
    ]

    tiles = add_tiles_footprint(tiles)
    all_tiles = xr.concat(tiles, dim="tile")
    
    all_tiles["origin_point"].attrs["comment"] = "Origin input point"
    all_tiles["tile_footprint"].attrs["comment"] = "Footprint of the tile"
    all_tiles["lon_centroid"].attrs["comment"] = "Longitude of the tile footprint's centroid"
    all_tiles["lat_centroid"].attrs["comment"] = "Latitude of the tile footprint's centroid"

    return dataset, all_tiles
