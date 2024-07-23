import matplotlib.pyplot as plt
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
import shapely
import os
import numpy as np
import logging
from datetime import datetime
from xsarsea.windspeed.models import get_model


def sigma0_detrend(sigma0, inc_angle, wind_speed_gmf=10., wind_dir_gmf=45., model='gmf_cmodifr2', line=10):
    """
    Detrend sigma0 using a given wind speed model.

    Parameters:
    - sigma0 (xarray.DataArray): Sigma0 data.
    - inc_angle (xarray.DataArray): Incidence angle data.
    - wind_speed_gmf (float): Wind speed for the GMF model.
    - wind_dir_gmf (float): Wind direction for the GMF model.
    - model (str): GMF model to use.
    - line (int): Line to use for sampling.

    Returns:
    - detrended (xarray.DataArray): Detrended sigma0 data.
    """
    model = get_model(model)
    try:
        sigma0_gmf_sample = inc_angle.isel(line=line).map_blocks(
            model, (wind_speed_gmf, wind_dir_gmf),
            template=inc_angle.isel(line=line),
            kwargs={'broadcast': True}
        )
    except AttributeError:
        sigma0_gmf_sample = model(inc_angle.isel(line=line), wind_speed_gmf, wind_dir_gmf, broadcast=True)

    gmf_ratio_sample = sigma0_gmf_sample / np.nanmean(sigma0_gmf_sample)
    detrended = sigma0 / gmf_ratio_sample.broadcast_like(sigma0)
    detrended.attrs['comment'] = f'detrended with model {model.name}'

    return detrended


def add_tiles_footprint(tiles):
    """
    Add footprint information to each tile in a list of tiles.

    Parameters:
    - tiles (list): List of tiles data.

    Returns:
    - tiles_with_footprint (list): List of tiles data with footprint information added.
    """
    if not isinstance(tiles, list):
        raise ValueError("tiles must be a list of tiles data.")
    tiles_with_footprint = []
    for tile in tqdm(tiles, desc='Adding footprints'):
        footprint_dict = {}
        for ll in ['longitude', 'latitude']:
            footprint_dict[ll] = [
                tile[ll].isel(tile_line=a, tile_sample=x).values for a, x in
                [(0, 0), (0, -1), (-1, -1), (-1, 0)]
            ]
        corners = list(zip(footprint_dict['longitude'], footprint_dict['latitude']))
        tile_footprint = Polygon(corners)
        centroids = tile_footprint.centroid
        tiles_with_footprint.append(
            tile.assign(tile_footprint=str(tile_footprint), lon_centroid=centroids.x, lat_centroid=centroids.y))

    return tiles_with_footprint


def save_tile(tiles, save_dir):
    """
    Saves radar or SAR tiles to NetCDF files.

    Parameters:
    - tiles (xarray.Dataset): The radar or SAR tiles dataset.
    - save_dir (str): Directory where the tiles should be saved.
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


def plot_cartopy_data(ds=None, tiles=None, polarization='VV', file_name='map'):
    """
    Plot SAR dataset or tiles overlaid on a map using Cartopy.

    Parameters:
    - ds (xarray.Dataset): SAR dataset.
    - tiles (list of xarray.DataArray): List of tiles.
    - polarization (str): Polarization to select from dataset or tiles.
    - file_name (str): Name of the output plot file.

    Returns:
    - None
    """
    if ds is None and tiles is None:
        raise ValueError("Either 'ds' or 'tiles' must be provided.")
    if tiles is not None and not isinstance(tiles, list):
        raise ValueError("'tiles' must be a list of tiles.")

    plt.figure(figsize=(10, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())

    if ds is not None:
        ds_p = ds.sel(pol=polarization)
        ax.pcolormesh(ds_p.longitude.data, ds_p.latitude.data, ds_p.sigma0.data, transform=ccrs.PlateCarree(),
                      cmap='viridis', zorder=1)

    if tiles is not None:
        for tile in tiles:
            img = tile.sel(pol=polarization).sigma0
            lon = tile.sel(pol=polarization).longitude
            lat = tile.sel(pol=polarization).latitude
            ax.pcolormesh(lon.data, lat.data, img.data, transform=ccrs.PlateCarree(), cmap='gray',
                          zorder=1)  # , vmin=0, vmax=0.02

            poly = shapely.wkt.loads(str(tile.tile_footprint.values))
            ax.plot(*poly.exterior.xy, '-b', linewidth=0.5, label='Patches footprint', zorder=1)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)

    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title(file_name)
    plt.show()
