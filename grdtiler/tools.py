from tqdm import tqdm
from shapely.geometry import Polygon
import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# def add_tiles_footprint(tiles):
#     """
#     Add footprint information to each tile in a list of tiles.

#     Args:
#         tiles (list): List of tile datasets.

#     Returns:
#         List[xr.Dataset]: List of tile datasets with footprint information added.

#     Raises:
#         ValueError: If the input is not a list or if any tile is missing required coordinates.
#     """
#     if not isinstance(tiles, list):
#         raise ValueError("tiles must be a list of tiles data.")
#     tiles_with_footprint = []
#     for tile in tqdm(tiles, desc='Adding footprints'):
#         footprint_dict = {}
#         for ll in ['longitude', 'latitude']:
#             footprint_dict[ll] = [
#                 tile[ll].isel(tile_line=a, tile_sample=x).values for a, x in
#                 [(0, 0), (0, -1), (-1, -1), (-1, 0)]
#             ]
#         corners = list(zip(footprint_dict['longitude'], footprint_dict['latitude']))
#         tile_footprint = Polygon(corners)
#         centroids = tile_footprint.centroid
#         tiles_with_footprint.append(
#             tile.assign(tile_footprint=str(tile_footprint), lon_centroid=centroids.x, lat_centroid=centroids.y))

#     return tiles_with_footprint

def process_single_tile(tile):
    """Process a single tile to add footprint information."""
    # Get corner coordinates using numpy operations instead of multiple isel calls
    corners_idx = [(0, 0), (0, -1), (-1, -1), (-1, 0)]
    
    # Extract coordinates all at once
    lons = tile['longitude'].values
    lats = tile['latitude'].values
    
    # Get corners using direct numpy indexing
    corner_coords = [
        (lons[i, j], lats[i, j])
        for i, j in corners_idx
    ]
    
    # Create polygon and get centroid
    tile_footprint = Polygon(corner_coords)
    centroids = tile_footprint.centroid
    
    # Return new tile with added attributes
    return tile.assign(
        tile_footprint=str(tile_footprint),
        lon_centroid=centroids.x,
        lat_centroid=centroids.y
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
    year = datetime.strptime(tiles.start_date, '%Y-%m-%d %H:%M:%S.%f').year
    day = datetime.strptime(tiles.start_date, '%Y-%m-%d %H:%M:%S.%f').timetuple().tm_yday
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

    if 'mean_wind_direction' in tiles.variables:
        save_name = filename.replace('GRDM', 'WDR').replace('GRDH', 'WDR').replace('GRD', 'WDR').replace('SGF', 'WDR')
        start_date = datetime.strptime(tiles.start_date, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y%m%dT%H%M%S')
        stop_date = datetime.strptime(tiles.stop_date, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y%m%dT%H%M%S')
        if 'S1' in filename:
            save_filename = (f"{save_name}/{safe[0]}-{tiles.swath.lower()}-wdr-{polarization[0].lower()}"
                             f"-{polarization[1].lower()}-{'-'.join(safe[4:-1])}.nc")
        elif 'RCM' in filename or 'RS2' in filename:
            save_filename = (f"{save_name}/{safe[0]}-{tiles.swath.lower()}-wdr-{polarization[0].lower()}"
                             f"-{polarization[1].lower()}-{start_date}-{stop_date}-{'-'.join(safe[5:7])}.nc")

    else:
        save_name = filename.replace('GRDM', 'TIL').replace('GRDH', 'TIL').replace('GRD', 'TIL').replace('SGF', 'WDR')
        start_date = datetime.strptime(tiles.start_date, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y%m%dT%H%M%S')
        stop_date = datetime.strptime(tiles.stop_date, '%Y-%m-%d %H:%M:%S.%f').strftime('%Y%m%dT%H%M%S')
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
            tiles.to_netcdf(save_path, mode='w', format='NETCDF4')
        except Exception as e:
            logging.info(f"Error saving tiles to {save_path}. Error: {e}")
    else:
        logging.info(f"This file {save_path} already exists.")

