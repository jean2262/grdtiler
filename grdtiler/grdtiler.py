import numpy as np
import xsar
import os
from tqdm import tqdm
import xbatcher
from datetime import datetime
import logging
from xsarslc.tools import xtiling, get_tiles

def tiling_prod(path, nperseg, resolution=None, noverlap=0, centering=False, side='left', tiling_mod='tiling', save_tiles=False, save_dir='.'):
    """
    Perform tiling on SAR data based on the specified parameters.

    Parameters:
        path (str): Path to the SAR dataset.
        nperseg (int or dict): Size of the subsets for tiling. If int, same size is used for both lines and samples.
            If dict, it should contain 'line' and 'sample' keys specifying separate sizes.
        resolution (str, optional): Resolution of the dataset (in meters). Default is None.
        noverlap (int or dict, optional): Overlap size between adjacent subsets. Default is 0.
            If int, same overlap is used for both lines and samples. If dict, separate overlaps can be specified.
        centering (bool, optional): Whether subsets should be centered on the dataset. Default is False.
        side (str, optional): Side to center on if centering is True. Can be 'left' or 'right'. Default is 'left'.
        tiling_mod (str, optional): Tiling module to use. Can be 'tiling' (custom tiling function), 'xtiling' (using xsarslc package),
            or 'xbatcher' (using xbatcher package). Default is 'tiling'.
        save_tiles (bool, optional): Whether to save the extracted tiles. Default is False.
        save_dir (str): Saving directory.

    Returns:
        tuple: A tuple containing the dataset and extracted tiles.
    """

    if 'GRD' not in path and 'RS2' not in path and 'RMC' not in path and 'RCM3' not in path:
        raise ValueError("This function can only tile datasets with types 'GRD', 'RS2', 'RMC', or 'RCM3'.")

    logging.info('Start tiling...')

    dataset = xsar.open_dataset(path, resolution)

    if tiling_mod == 'xtiling':
        tiles_index = xtiling(ds=dataset, nperseg=nperseg, noverlap=noverlap, centering=centering, side=side)
        tiles = get_tiles(ds=dataset, tiles_index=tiles_index)

    elif tiling_mod == 'xbatcher':
        if centering:
            print('Warning: xbatcher tiling module does not support centering option.')
        dataset = dataset.rename({'sample': 'column'})
        if not isinstance(nperseg, dict):
            nperseg = {'line': nperseg, 'column': nperseg}
        if 'sample' in nperseg:
            nperseg['column'] = nperseg.pop('sample')
        if not isinstance(noverlap, dict):
            noverlap = {'line': noverlap, 'column': noverlap}
        xb_tiles = xbatcher.BatchGenerator(ds=dataset, input_dims=nperseg, input_overlap=noverlap)
        tiles = [tile.rename({'column': 'sample'}) for tile in xb_tiles]

    elif tiling_mod == 'tiling':
        tiles = tiling(dataset=dataset, subset_size=nperseg, centering=centering, side=side, noverlap=noverlap)

    else:
        raise ValueError("Invalid tiling module. Please choose one of 'tiling', 'xtiling', or 'xbatcher'.")

    logging.info('Done tiling...')

    if save_tiles:
        save_tile(tiles, resolution, save_dir)

    return dataset, tiles


def tiling_by_point(path ,posting_loc, posting_box_size=0, resolution=None, save_tiles=False, save_dir='.'):
    """
    Extract tiles centered around a specified point from a SAR dataset.

    Parameters:
        path (str): Path to the SAR dataset.
        posting_loc (tuple): Coordinates (longitude, latitude) of the point.
            Default is None, which will use the center of the dataset's spatial extent.
        posting_box_size (float): Size of the box centered around the point (in meters). Default is zero.
            Default is None, which will use the maximum size that fits within the dataset's spatial extent.
        resolution (str, optional): Resolution of the dataset (in meters). Default is None.
        save_tiles (bool, optional): Whether to save the extracted tiles. Default is False.
        save_dir (str): Saving directory.

    Returns:
        xarray.Dataset: Extracted tiles centered around the specified point.
    """

    if 'GRD' not in path and 'RS2' not in path and 'RMC' not in path and 'RCM3' not in path:
        raise ValueError("This function can only tile datasets with types 'GRD', 'RS2', 'RMC', or 'RCM3'.")

    logging.info('Start tiling...')

    sar_ds = xsar.Sentinel1Dataset(path, resolution)

    # If posting_loc is not provided, use the center of the dataset's spatial extent
    if posting_loc is None:
        raise ValueError('Posting localisation must be given.')

    lon, lat = posting_loc
    point_coords = sar_ds.ll2coords(lon, lat)
    dist = {'line' : int(np.round(posting_box_size / 2 / 10.)), 'sample': int(np.round(posting_box_size / 2 / 10.))}
    tiles = sar_ds.dataset.sel(line=slice(point_coords[0] - dist['line'], point_coords[0] + dist['line']), sample=slice(point_coords[1] - dist['sample'], point_coords[1] + dist['sample']))

    logging.info('Done tiling...')

    if save_tiles:
        save_tile(tiles, resolution, save_dir)

    return tiles



def tiling(dataset, subset_size, noverlap, centering, side):
    """
    Generate overlapping or non-overlapping subsets from a dataset.

    Parameters:
        dataset (xarray.Dataset or xarray.DataArray): Input dataset.
        subset_size (int or dict): Size of the subsets. If int, same size is used for both lines and samples.
            If dict, it should contain 'line' and 'sample' keys specifying separate sizes.
        noverlap (int or dict, optional): Overlap size between adjacent subsets. Default is 0.
            If int, same overlap is used for both lines and samples. If dict, separate overlaps can be specified.
        centering (bool, optional): Whether subsets should be centered on the dataset. Default is False.
        side (str, optional): Side to center on if centering is True. Can be 'left' or 'right'. Default is 'left'.

    Returns:
        list: List of subsets.
    """
    subsets = []

    # Calculate tile size and overlap
    tile_line_size, tile_sample_size = (subset_size.get('line', 1), subset_size.get('sample', 1)) if isinstance \
        (subset_size, dict) else (subset_size, subset_size)
    line_overlap, sample_overlap = (noverlap.get('line', 0), noverlap.get('sample', 0)) if isinstance(noverlap, dict) else \
    (noverlap, noverlap)

    total_lines, total_samples = dataset.sizes['line'], dataset.sizes['sample']
    mask = dataset

    # Centering logic
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

    # Input validation
    if noverlap >= min(tile_line_size, tile_sample_size):
        raise ValueError('Overlap size must be less than tile size')

    # Calculate step sizes
    step_line = tile_line_size - noverlap
    step_sample = tile_sample_size - noverlap

    # Generate subsets
    for line_start in range(0, total_lines - tile_line_size + 1, step_line):
        for sample_start in range(0, total_samples - tile_sample_size + 1, step_sample):
            subset = mask.isel(line=slice(line_start, line_start + tile_line_size),
                               sample=slice(sample_start, sample_start + tile_sample_size))
            subsets.append(subset)

    return subsets


def save_tile(tiles, resolution, save_dir):
    """
    Save tiles into NetCDF files.

    Parameters:
        tiles (list): A list of tiles to be saved.
        resolution (str): Resolution of the tiles.
        save_dir (str): Saving directory.

    Returns:
        None
    """
    base_path = save_dir
    filename = os.path.basename(tiles[0].name).split(".SAFE")[0]
    year = datetime.strptime(tiles[0].start_date, '%Y-%m-%d %H:%M:%S.%f').year
    day = datetime.strptime(tiles[0].start_date, '%Y-%m-%d %H:%M:%S.%f').timetuple().tm_yday

    for i, tile in tqdm(enumerate(tiles), total=len(tiles), desc='Saving'):
        for pol in tile.pol.values:
            mode = tile.sel(pol=pol).swath
            size_line = tile.sel(pol=pol).line.size
            size_sample = tile.sel(pol=pol).sample.size
            ds_dir = f"{base_path}/data_nc/{mode}/size_{size_line}_{size_sample}/res_{resolution}/{year}/{day}/{filename}/{pol}"

            try:
                os.makedirs(ds_dir, exist_ok=True)
            except OSError as e:
                print(f"Error creating directory: {ds_dir}. Error: {e}")
                continue

            for attr in ['footprint', 'multidataset', 'specialHandlingRequired']:
                if attr in tile.attrs:
                    tile.attrs[attr] = str(tile.attrs[attr])

            if 'gcps' in tile.spatial_ref.attrs:
                tile.spatial_ref.attrs.pop('gcps')

            save_path = os.path.join(f"{ds_dir}/{filename}_{pol}_{i}.nc")

            try:
                tile.to_netcdf(save_path, mode='w', format='NETCDF4')
            except Exception as e:
                print(f"Error saving tile to {save_path}. Error: {e}")


