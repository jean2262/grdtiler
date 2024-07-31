import logging
import numpy as np
import xsar
import xarray as xr
from tqdm import tqdm
from grdtiler.tools import sigma0_detrend, add_tiles_footprint, save_tile


# Function to tile SAR dataset
def tiling_prod(
    path,
    tile_size,
    resolution=None,
    detrend=True,
    noverlap=0,
    centering=False,
    side='left',
    save=False,
    save_dir='.',
    to_keep_var=None
):
    """
    Tiles a radar or SAR dataset.

    Args:
        path (str): Path to the radar or SAR dataset.
        tile_size (int or dict): Size of each tile in pixels (height, width).
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
        ValueError: If the dataset type is not 'GRD', 'RS2', or 'RCM'.
    """
    logging.info('Start tiling...')

    supported_datasets = {'GRD', 'RS2', 'RCM'}
    dataset_type = next((dt for dt in supported_datasets if dt in path), None)

    if not dataset_type:
        raise ValueError(f"Unsupported dataset type. Supported types are: {', '.join(supported_datasets)}")

    dataset = xsar.open_dataset(path, resolution)

    dataset, nperseg = tile_normalize(dataset, tile_size, resolution, detrend, to_keep_var)
    tiles = tiling(dataset=dataset, tile_size=nperseg, noverlap=noverlap, centering=centering, side=side)

    logging.info('Done tiling...')

    if save:
        save_tile(tiles, save_dir)

    return dataset, tiles


# Function to normalize SAR dataset for tiling
def tile_normalize(
    dataset,
    tile_size,
    resolution,
    detrend=True,
    to_keep_var=None
):
    """
    Normalize a radar or SAR dataset for tiling.

    Args:
        dataset (xr.Dataset): The radar or SAR dataset.
        tile_size (int or dict): Size of each tile in meters. If int, represents size along both dimensions.
            If dict, should have keys 'line' and/or 'sample' indicating size along each dimension.
        resolution (str): Resolution of the dataset in meters.
        detrend (bool, optional): Whether to detrend the image. Defaults to True.
        to_keep_var (list, optional): Variables to keep in the dataset. Defaults to None.

    Returns:
        The normalized radar or SAR dataset.
        Number of pixels per segment for tiling (int or dict with 'line' and 'sample' keys).
    """
    default_vars = ['longitude', 'latitude']
    if to_keep_var:
        to_keep_var.extend(default_vars)
    else:
        to_keep_var = ['sigma0', 'land_mask', 'ground_heading', 'incidence', 'nesz']
        to_keep_var.extend(default_vars)

    resolution_value = int(resolution.split('m')[0]) if resolution else 1

    if isinstance(tile_size, dict):
        tile_line_size = tile_size.get('line', 1)
        tile_sample_size = tile_size.get('sample', 1)
        nperseg = {'line': tile_line_size // resolution_value, 'sample': tile_sample_size // resolution_value}
        dataset.attrs['tile_size'] = f'{tile_line_size}m*{tile_sample_size}m (line * sample)'
    else:
        nperseg = tile_size // resolution_value
        dataset.attrs['tile_size'] = f'{tile_size}m*{tile_size}m (line * sample)'

    dataset.attrs.update({
        'resolution': resolution,
        'polarizations': dataset.attrs['pols'],
        'processing_level': dataset.attrs['product'],
        'main_footprint': dataset.attrs['footprint']
    })

    if 'platform_heading' in dataset.attrs:
        dataset.attrs['platform_heading(degree)'] = dataset.attrs['platform_heading']

    if detrend:
        dataset['sigma0_no_nan'] = xr.where(dataset['land_mask'], np.nan, dataset['sigma0'])
        dataset['sigma0_detrend'] = sigma0_detrend(dataset['sigma0_no_nan'], dataset['incidence'], line=10)
        to_keep_var.append('sigma0_detrend')

    if 'longitude' in dataset.variables and 'latitude' in dataset.variables:
        dataset['sigma0'] = dataset['sigma0'].transpose(*dataset['sigma0'].dims)

    dataset = dataset.drop_vars(set(dataset.data_vars) - set(to_keep_var))

    attributes_to_remove = {'name', 'multidataset', 'product', 'pols', 'footprint', 'platform_heading'}
    dataset.attrs = {key: value for key, value in dataset.attrs.items() if key not in attributes_to_remove}

    if 'spatial_ref' in dataset.coords and 'gcps' in dataset.spatial_ref.attrs:
        dataset.spatial_ref.attrs.pop('gcps')

    return dataset, nperseg


# Function to generate tiles from SAR dataset
def tiling(
    dataset,
    tile_size,
    noverlap,
    centering,
    side
):
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
    tile_line_size, tile_sample_size = (tile_size.get('line', 1), tile_size.get('sample', 1)) \
        if isinstance(tile_size, dict) else (tile_size, tile_size)
    line_overlap, sample_overlap = (noverlap.get('line', 0), noverlap.get('sample', 0)) \
        if isinstance(noverlap, dict) else (noverlap, noverlap)

    total_lines, total_samples = dataset.sizes['line'], dataset.sizes['sample']
    mask = dataset

    if noverlap >= min(tile_line_size, tile_sample_size):
        raise ValueError('Overlap size must be less than tile size')

    if centering:
        complete_segments_line = (total_lines - tile_line_size) // (tile_line_size - line_overlap) + 1
        mask_size_line = complete_segments_line * tile_line_size - (complete_segments_line - 1) * line_overlap

        complete_segments_sample = (total_samples - tile_sample_size) // (tile_sample_size - sample_overlap) + 1
        mask_size_sample = complete_segments_sample * tile_sample_size - (complete_segments_sample - 1) * sample_overlap

        if side == 'right':
            start_line = (total_lines // 2) - (mask_size_line // 2)
            start_sample = (total_samples // 2) - (mask_size_sample // 2)
        else:
            start_line = (total_lines // 2) + (total_lines % 2) - (mask_size_line // 2)
            start_sample = (total_samples // 2) + (total_samples % 2) - (mask_size_sample // 2)

        mask = dataset.isel(line=slice(start_line, start_line + mask_size_line),
                            sample=slice(start_sample, start_sample + mask_size_sample))

    step_line = tile_line_size - noverlap
    step_sample = tile_sample_size - noverlap

    for line_start in tqdm(range(0, total_lines - tile_line_size + 1, step_line), desc='Tiling'):
        for sample_start in range(0, total_samples - tile_sample_size + 1, step_sample):
            subset = mask.isel(line=slice(line_start, line_start + tile_line_size),
                               sample=slice(sample_start, sample_start + tile_sample_size))
            if len(subset['line'].values) == tile_line_size and len(subset['sample'].values) == tile_sample_size:
                tiles.append(
                    subset.drop_indexes(['line', 'sample']).rename_dims({'line': 'tile_line', 'sample': 'tile_sample'}))
    if not tiles:
        raise ValueError('No tiles')

    tiles_with_footprint = add_tiles_footprint(tiles)
    all_tiles = xr.concat(tiles_with_footprint, dim='tile')
    all_tiles['tile_footprint'].attrs['comment'] = 'Footprint of the tile'
    all_tiles['lon_centroid'].attrs['comment'] = 'Longitude of the tile footprint\'s centroid'
    all_tiles['lat_centroid'].attrs['comment'] = 'Latitude of the tile footprint\'s centroid'

    return all_tiles


# Function to tile a radar or SAR dataset around specified points
def tiling_by_point(
    path,
    posting_loc,
    tile_size,
    resolution=None,
    detrend=True,
    save=False,
    save_dir='.',
    to_keep_var=None
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

    logging.info('Start tiling...')

    if 'GRD' in path and 'RS2' not in path and 'RCM' not in path:
        sar_dm = xsar.Sentinel1Meta(path)
        sar_ds = xsar.Sentinel1Dataset(sar_dm, resolution)
    elif 'RS2' in path:
        sar_ds = xsar.RadarSat2Dataset(path, resolution)
    elif 'RCM' in path:
        sar_ds = xsar.RcmDataset(path, resolution)
    else:
        raise ValueError("This function can only tile datasets with types 'GRD', 'RS2', 'RMC', or 'RCM3'.")

    tiles = []
    dataset = sar_ds.dataset
    dataset, _ = tile_normalize(dataset, tile_size, resolution, detrend, to_keep_var)
    for point in tqdm(posting_loc, desc='Tiling'):
        if point is None:
            raise ValueError(f"Invalid posting location: {posting_loc}")

        lon, lat = point.x, point.y
        point_coords = sar_ds.ll2coords(lon, lat)
        if np.isnan(point_coords).any():
            raise ValueError(f"Choose a point inside the footprint: {sar_ds.footprint}")

        if 'GRD' in path and 'RS2' not in path and 'RCM' not in path:
            dist = {'line': int(np.round(tile_size / 2 / sar_dm.pixel_line_m)),
                    'sample': int(np.round(tile_size / 2 / sar_dm.pixel_sample_m))}
        else:
            dist = {'line': int(np.round(tile_size / 2 / dataset.pixel_line_m)),
                    'sample': int(np.round(tile_size / 2 / dataset.pixel_sample_m))}

        tile = dataset.sel(line=slice(point_coords[0] - dist['line'], point_coords[0] + dist['line'] - 1),
                           sample=slice(point_coords[1] - dist['sample'], point_coords[1] + dist['sample'] - 1))

        tiles.append(tile.drop_indexes(['line', 'sample']).rename_dims({'line': 'tile_line', 'sample': 'tile_sample'}))

    logging.info('Done tiling...')

    tiles = add_tiles_footprint(tiles)
    all_tiles = xr.concat(tiles, dim='tile')
    all_tiles['tile_footprint'].attrs['comment'] = 'Footprint of the tile'
    all_tiles['lon_centroid'].attrs['comment'] = 'Longitude of the tile footprint\'s centroid'
    all_tiles['lat_centroid'].attrs['comment'] = 'Latitude of the tile footprint\'s centroid'

    if save:
        save_tile(all_tiles, save_dir)

    return dataset, all_tiles
