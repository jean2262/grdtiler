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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PIXELSPACING = {
    "S1_IW_GRDH": 10,
    "S1_IW_GRDM": 40,
    "S1_EW_GRDH": 40,
    "S1_EW_GRDV": 40,
}

# Function to tile SAR dataset
def tiling_prod(
    path: str|xr.Dataset,
    tile_size,
    resolution=None,
    detrend=True,
    noverlap=0,
    centering=False,
    side="left",
    save=False,
    save_dir=".",
    to_keep_var=None,
    add_footprint=True,
    config_file="/home1/datawork/jrmiadan/project/grdtiler/grdtiler/config.yaml",
):

    """
    Tiles a radar or SAR dataset.

    Args:
        path (str): Path to the radar or SAR dataset.
        tile_size (int or dict): Size of each tile in meters (height, width).
        resolution (str , optional): Resolution of the dataset. Defaults to None.
        detrend (bool, optional): Whether to detrend the image. Defaults to True.
        noverlap (int, optional): Number of pixels to overlap between adjacent tiles. Defaults to 0.
        centering (bool, optional): If True, centers the tiles within the dataset. Defaults to False.
        side (str, optional): Side of the dataset from which tiling starts ('left' or 'right'). Defaults to 'left'.
        save (bool, optional): If True, saves the tiled dataset. Defaults to False.
        save_dir (str, optional): Directory where the tiled dataset should be saved. Defaults to current directory.
        to_keep_var (list, optional): Variables to keep in the dataset. Defaults to None.

    Returns:
        dataset: The radar or SAR dataset.
        tiles: The tiled radar or SAR dataset.

    Raises:
        ValueError: If the dataset type is not 'S1', 'RS2', or 'RCM'.
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

def load_gmf_model(filename, config_file, pol):
    """
    Load a GMF model according to the filename and polarisation.

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 data.
    config_file : str, optional
        The path to the configuration file. Defaults to "./config.yaml".
    pol : str
        The polarisation of the data. One of "HH", "HV", "VV", "VH".

    Returns
    -------
    gmf_model : str
        The name of the GMF model to use.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if filename.upper().startswith("S1"):
        mission = "S1"
    elif filename.upper().startswith("RS2"):
        mission = "RS2"
    elif filename.upper().startswith("RCM"):
        mission = "RCM"
    else:
        raise ValueError("Only support S1, RCM and RS2")
        
    gmf_base_path = config["gmf_base_path"]

    if pol == "HH":
        xsarsea.windspeed.register_nc_luts(gmf_base_path)
        gmf_model = config['gmf_models'][mission]['GMF_HH_NAME']
        
    elif pol == "HV":
        xsarsea.windspeed.register_nc_luts(gmf_base_path)
        gmf_model = config['gmf_models'][mission]['GMF_HV_NAME']
        
    elif pol == "VV":
        gmf_model = config['gmf_models'][mission]['GMF_VV_NAME']
        
    elif pol == "VH":
        gmf_model = config['gmf_models'][mission]['GMF_VH_NAME']

    return gmf_model

def move_valid_line_to_zero(inc_angle):
    nan_mask = np.isnan(inc_angle.values)
    nan_counts = nan_mask.sum(axis=tuple(range(1, nan_mask.ndim)))
    valid_lines = np.where(nan_counts == 0)[0]
    if valid_lines.size > 0:
        first_valid_line = valid_lines[0]
    else:
        first_valid_line = np.argmin(nan_counts) 

    sliced = inc_angle.isel(line=slice(first_valid_line, None))
    return sliced.assign_coords(line=np.arange(sliced.sizes['line']))

def tile_normalize(dataset, tile_size, resolution, noverlap=0, detrend=True, to_keep_var=None, config_file="./config.yml"):
    """
    Normalize a radar or SAR dataset for tiling.

    Args:
        tile_size (int or dict): Size of each tile in meters. If int, represents size along both dimensions.
            If dict, should have keys 'line' and/or 'sample' indicating size along each dimension.
        resolution (str): Resolution of the dataset in meters.
        detrend (bool, optional): Whether to detrend the image. Defaults to True.
        to_keep_var (list, optional): Variables to keep in the dataset. Defaults to None.

    Returns:
        The normalized radar or SAR dataset.
        Number of pixels per segment for tiling (int or dict with 'line' and 'sample' keys).
    """
    default_vars = ["longitude", "latitude"]
    
    # logger.info(f"to_keep_var: {to_keep_var}")
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

    # if "platform_heading" in dataset.attrs:
    #     dataset.attrs["platform_heading(degree)"] = dataset.attrs["platform_heading"]
    
    if detrend:
        dataset["sigma0_no_nan"] = xr.where(
            dataset["land_mask"], np.nan, dataset["sigma0"]
        )
        filename = dataset.attrs["safe"] if "safe" in dataset.attrs else dataset.attrs["name"]
        if ":" in filename:
            filename = Path(filename.split(":")[1]).name
        sigma0_detrends = []
        for pol in dataset.pol.values:
            gmf_model = load_gmf_model(filename=filename, config_file=config_file, pol=pol)
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

    if "spatial_ref" in dataset.coords and "gcps" in dataset.spatial_ref.attrs:
        dataset.spatial_ref.attrs.pop("gcps")

    return dataset, nperseg


def tiling(dataset, tile_size, noverlap, centering, side, add_footprint=True):
    """
    Generates tiles from a radar or SAR (Synthetic Aperture Radar) dataset.

    Args:
        dataset (xr.Dataset): The radar or SAR dataset.
        tile_size (int or dict): Size of each tile in pixels (height, width).
            If int, it represents the height and the width of the tile.
            If dict, it should have keys 'line' and 'sample' indicating size along each dimension.
        noverlap (int or dict): Number of pixels to overlap between adjacent tiles.
            If int, it's applied to both dimensions.
            If dict, it should have keys 'line' and 'sample' indicating overlap along each dimension.
        centering (bool): If True, centers the tiles within the dataset.
        side (str): Side of the dataset from which tiling starts. Must be 'left' or 'right'.

    Returns:
        xr.Dataset: A concatenated xarray dataset containing all generated tiles.

    Raises:
        ValueError: If noverlap is greater than or equal to tile size, or if no tiles are generated.
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

    if noverlap >= min(tile_line_size, tile_sample_size):
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

    step_line = tile_line_size - noverlap
    step_sample = tile_sample_size - noverlap

    for line_start in tqdm(
        range(0, total_lines - tile_line_size + 1, step_line), desc="Tiling"
    ):
        for sample_start in range(0, total_samples - tile_sample_size + 1, step_sample):
            subset = mask.isel(
                line=slice(line_start, line_start + tile_line_size),
                sample=slice(sample_start, sample_start + tile_sample_size),
            )
            if (
                len(subset["line"].values) == tile_line_size
                and len(subset["sample"].values) == tile_sample_size
            ):
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

    lon = ds.longitude.compute().values
    lat = ds.latitude.compute().values

    dist2 = (lon - lon0)**2 + (lat - lat0)**2

    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)

    line = ds.line.values[iy]
    sample = ds.sample.values[ix]

    return (line, sample)

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
    config_file="/home1/datawork/jrmiadan/project/grdtiler/grdtiler/config.yaml",
):
    """
    Tiles a radar or SAR dataset around specified points.

    Args:
        path (str): Path to the radar or SAR dataset.
        posting_loc (list): Points around which to tile the dataset.
        tile_size (int): Size of the box (in meters) to be tiled around each point.
        resolution (str, optional): Resolution of the dataset. Defaults to None.
        detrend (bool, optional): Make detrend image. Defaults to True.
        save (bool, optional): If True, saves the tiled dataset. Defaults to False.
        save_dir (str, optional): Directory where the tiled dataset should be saved. Defaults to current directory.
        to_keep_var (list, optional): Variables to keep in the dataset. Defaults to None.

    Returns:
        dataset: The radar or SAR dataset.
        all_tiles (xarray.Dataset): A concatenated xarray dataset containing all generated tiles.

    Raises:
        ValueError: If the dataset type is unsupported or if an invalid posting location is provided.
    """
    dataset = load_dataset(path, resolution=resolution)
    
    logger.info("Start tiling...")
    
    tiles = []
    filename = dataset.attrs["safe"] if "safe" in dataset.attrs else dataset.attrs["name"]
    if ":" in filename:
        filename = Path(filename.split(":")[1]).name
    
    fn = "_".join([filename[:2]] + filename.split("_")[1:3])
    footprint = dataset.attrs["footprint"]
    footprint = wkt.loads(footprint) if isinstance(footprint, str) else footprint
    dataset, nperseg = tile_normalize(
        dataset=dataset, tile_size=tile_size, resolution=resolution, detrend=detrend, to_keep_var=to_keep_var, config_file=config_file
    )
    for i, point in tqdm(enumerate(posting_loc), total=len(posting_loc), desc="Tiling"):
        if point is None:
            raise ValueError(f"Invalid posting location: {posting_loc}")


        if not footprint.contains(point):
            logging.warning(f"Skipping {point} as it is outside the footprint.")
            continue
        # point_coords = sar_ds.ll2coords(lon, lat)
        point_coords = find_pixel(dataset, point.x, point.y)
        
        if np.isnan(point_coords).any():
            logging.warning(
                f"Choose a point inside the footprint: {footprint}, for: {point}"
            )
            continue
            # raise ValueError(f"Choose a point inside the footprint: {sar_ds.footprint}")

        if filename.startswith("S1"):
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
                logging.warning(f"Error on tile size {tile.sizes}, for {point}")
                continue
        else:
            if (
                not tile.sizes["line"] == nperseg["line"]
                or not tile.sizes["sample"] == nperseg["sample"]
            ):
                logging.warning(f"Error on tile size {tile.sizes}, for {point}")
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
        print("no tiles")
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
    config_file="/home1/datawork/jrmiadan/project/grdtiler/grdtiler/config.yaml",
):
    """_summary_

    Args:
        path (_type_): _description_
        tile_size (_type_): _description_
        resolution (_type_, optional): _description_. Defaults to None.
        detrend (bool, optional): _description_. Defaults to True.
        to_keep_var (_type_, optional): _description_. Defaults to None.
        config_file (str, optional): _description_. Defaults to "/home1/datawork/jrmiadan/project/grdtiler/grdtiler/config.yaml".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
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
    is_cross_antimeridian = crosses_antimeridian(dataset.main_footprint)
    
    if is_cross_antimeridian:
        correct_footprint = normalize_longitudes_to_360(dataset.main_footprint)
        point = correct_footprint.centroid
        if correct_footprint is None:
            logger.warning("Unable to normalize longitudes to 360 degrees.")
            return dataset, None
        
        line_mid = dataset.sizes["line"] // 2
        sample_mid = dataset.sizes["sample"] // 2

        tile = dataset.isel(
            line=slice(line_mid - (nperseg // 2), line_mid + (nperseg // 2)),
            sample=slice(sample_mid - (nperseg // 2), sample_mid + (nperseg // 2))
        )
    else:
        point = dataset.main_footprint.centroid
        lon, lat = point.x, point.y
        point_geom = Point(lon, lat)

        if not dataset.main_footprint.contains(point_geom):
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
        
        # Ajuste if tile is too small
        if tile.sizes["line"] < nperseg and tile.sizes["sample"] < nperseg:
            tile = dataset.sel(
                line=slice(
                    point_coords[0] - dist["line"] - 1, point_coords[0] + dist["line"] - 1
                ),
                sample=slice(
                    point_coords[1] - dist["sample"] - 1, point_coords[1] + dist["sample"] - 1
                ),
            )
        
        elif tile.sizes["line"] < nperseg and not tile.sizes["sample"] < nperseg:
            tile = dataset.sel(
                line=slice(
                    point_coords[0] - dist["line"] - 1, point_coords[0] + dist["line"] - 1
                ),
                sample=slice(
                    point_coords[1] - dist["sample"], point_coords[1] + dist["sample"] - 1
                ),
            )
            
        elif not tile.sizes["line"] < nperseg and tile.sizes["sample"] < nperseg:
            tile = dataset.sel(
                line=slice(
                    point_coords[0] - dist["line"], point_coords[0] + dist["line"] - 1
                ),
                sample=slice(
                    point_coords[1] - dist["sample"] - 1, point_coords[1] + dist["sample"] - 1
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

    if not tiles:
        logger.warning("No tiles generated")
        return dataset, None

    tiles = add_tiles_footprint(tiles)
    all_tiles = xr.concat(tiles, dim="tile")
    
    all_tiles["origin_point"].attrs["comment"] = "Point d'origine de l'entrée"
    all_tiles["tile_footprint"].attrs["comment"] = "Empreinte de la tuile"
    all_tiles["lon_centroid"].attrs["comment"] = "Longitude du centroïde de la tuile"
    all_tiles["lat_centroid"].attrs["comment"] = "Latitude du centroïde de la tuile"

    return dataset, all_tiles
