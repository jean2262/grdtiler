from tqdm import tqdm
from shapely.geometry import Polygon
import numpy as np
import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def process_single_tile(tile):
    """Process a single tile to add footprint information."""
    corners_idx = [(0, 0), (0, -1), (-1, -1), (-1, 0)]

    lons = tile['longitude'].values
    lats = tile['latitude'].values

    corner_coords = [(lons[i, j], lats[i, j]) for i, j in corners_idx]

    # Store the footprint in original [-180, 180] space.
    tile_footprint = Polygon(corner_coords)

    # Detect antimeridian crossing: any corner pair straddles ±180°.
    corner_lons = [c[0] for c in corner_coords]
    crosses = max(corner_lons) > 150 and min(corner_lons) < -150

    if crosses:
        # Normalize to [0, 360] so Shapely computes the centroid correctly.
        norm_coords = [((lon + 360) if lon < 0 else lon, lat) for lon, lat in corner_coords]
        centroid = Polygon(norm_coords).centroid
        # Convert centroid longitude back to [-180, 180].
        lon_c = centroid.x - 360 if centroid.x > 180 else centroid.x
        lat_c = centroid.y
    else:
        centroid = tile_footprint.centroid
        lon_c = centroid.x
        lat_c = centroid.y

    # Store as object dtype to prevent fixed-length string truncation in NetCDF4.
    # Antimeridian WKTs are longer (extra '-' sign per coord) and would be clipped
    # if numpy used a U{n} dtype sized by the shorter normal-tile footprints.
    return tile.assign(
        tile_footprint=np.array(str(tile_footprint), dtype=object),
        lon_centroid=lon_c,
        lat_centroid=lat_c,
    )

def add_tiles_footprint(tiles, max_workers=None):
    """
    Add footprint information to each tile in a list of tiles.
    
    Args:
        tiles (list): List of tile datasets.
        max_workers (int, optional): Maximum number of worker threads.
            Defaults to None (uses ThreadPoolExecutor default).
    
    Returns:
        List[xr.Dataset]: List of tile datasets with footprint information added.
    
    Raises:
        ValueError: If the input is not a list or if any tile is missing required coordinates.
    """
    if not isinstance(tiles, list):
        raise ValueError("tiles must be a list of tiles data.")
        
    # Process tiles in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress while processing in parallel
        tiles_with_footprint = list(
            tqdm(
                executor.map(process_single_tile, tiles),
                total=len(tiles),
                desc='Adding footprints'
            )
        )
    
    return tiles_with_footprint


def save_tile(tiles, save_dir):
    """
    Saves radar or SAR tiles to NetCDF files.

    Args:
        tiles (xr.Dataset): The radar or SAR tiles dataset.
        save_dir (str): Directory where the tiles should be saved.
    """
    base_path = save_dir
    start_dt = datetime.strptime(tiles.start_date, '%Y-%m-%d %H:%M:%S.%f')
    year = start_dt.year
    day = start_dt.timetuple().tm_yday
    tile_sizes = tiles.attrs['tile_size'].split(' ')[0].split('*')
    resolution = tiles.attrs['resolution']
    mode = tiles.swath

    tiles_dir = f"{base_path}/GRD/{mode}/size_{tile_sizes[0]}_{tile_sizes[1]}/res_{resolution}/{year}/{day}/"

    for attr in ['main_footprint', 'specialHandlingRequired']:
        if attr in tiles.attrs:
            tiles.attrs[attr] = str(tiles.attrs[attr])

    if 'satellite' in tiles.attrs:
        filename = os.path.basename(tiles.product_path)
        safe = filename.lower().split('_')
    else:
        filename = tiles.safe
        safe = filename.lower().split('_')

    polarization = tiles.polarizations.split(' ')

    start_date = start_dt.strftime('%Y%m%dT%H%M%S')
    stop_date = datetime.strptime(tiles.stop_date, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y%m%dT%H%M%S')

    if 'mean_wind_direction' in tiles.variables:
        save_name = filename.replace('GRDM', 'WDR').replace('GRDH', 'WDR').replace('GRD', 'WDR').replace('SGF', 'WDR')
        if 'S1' in filename:
            save_filename = (f"{save_name}/{safe[0]}-{tiles.swath.lower()}-wdr-{polarization[0].lower()}"
                             f"-{polarization[1].lower()}-{'-'.join(safe[4:-1])}.nc")
        elif 'RCM' in filename or 'RS2' in filename:
            save_filename = (f"{save_name}/{safe[0]}-{tiles.swath.lower()}-wdr-{polarization[0].lower()}"
                             f"-{polarization[1].lower()}-{start_date}-{stop_date}-{'-'.join(safe[5:7])}.nc")

    else:
        save_name = filename.replace('GRDM', 'TIL').replace('GRDH', 'TIL').replace('GRD', 'TIL').replace('SGF', 'WDR')
        if 'S1' in filename:
            save_filename = (f"{save_name}/{safe[0]}-{tiles.swath.lower()}-til-{polarization[0].lower()}"
                             f"-{polarization[1].lower()}-{'-'.join(safe[4:-1])}.nc")
        elif 'RCM' in filename or 'RS2' in filename:
            save_filename = (f"{save_name}/{safe[0]}-{tiles.swath.lower()}-til-{polarization[0].lower()}"
                             f"-{polarization[1].lower()}-{start_date}-{stop_date}-{'-'.join(safe[5:7])}.nc")

    os.makedirs(tiles_dir + save_name, exist_ok=True)
    save_path = os.path.join(tiles_dir, save_filename)
    if not os.path.exists(save_path):
        try:
            # Cast fixed-length unicode variables to object dtype so xarray writes
            # them as variable-length NetCDF4 strings, preventing WKT truncation.
            save_ds = tiles.copy()
            for var_name, da in save_ds.data_vars.items():
                if da.dtype.kind in ("U", "S"):
                    save_ds[var_name] = da.astype(object)
            save_ds.to_netcdf(save_path, mode='w', format='NETCDF4')
        except Exception as e:
            logging.info(f"Error saving tiles to {save_path}. Error: {e}")
    else:
        logging.info(f"This file {save_path} already exists.")

