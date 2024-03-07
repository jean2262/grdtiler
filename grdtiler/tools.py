import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv
hv.extension('bokeh')
from tqdm import tqdm
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon


def tiles_images(tiles, polarization):
    """
    Generate HoloViews Overlay of images from tiles data with specified polarization.

    Parameters:
    - tiles (list): A list of tiles data.
    - polarization (str): The polarization to select from each tile.

    Returns:
    - hv.Overlay: A HoloViews Overlay containing the images generated from tiles data.
    """
    if not isinstance(tiles, list):
        raise ValueError("tiles must be a list of tiles data.")

    if polarization not in tiles[0].pol.values:
        raise ValueError("Invalid polarization option. Please choose from the available polarizations.")

    obs_hmaps = []

    for i, tile in enumerate(tiles):
        bounds = (
            tile.line.min().values,
            tile.sample.min().values,
            tile.line.max().values,
            tile.sample.max().values
        )

        obs_hmap = hv.Image(np.rot90(tile.sigma0.sel(pol=polarization).values), bounds=bounds).opts(cmap='gray')
        obs_hmaps.append(obs_hmap)

    tiles_img = hv.Overlay(obs_hmaps).opts(width=800, height=800)

    return tiles_img


def get_tiles_footprint(tiles, polarization):
    """
    Extract the footprint of each tile and store it as a Polygon object.

    Parameters:
    - tiles (list): A list of tiles data.
    - polarization (str): The polarization to select from each tile.

    Returns:
    - list: A list of Polygon objects representing the footprint of each tile.
    """

    if not isinstance(tiles, list):
        raise ValueError("tiles must be a list of tiles data.")
    if polarization not in tiles[0].pol.values:
        raise ValueError("Invalid polarization option. Please choose from the available polarizations.")

    tiles_footprint = []
    for tile in tqdm(tiles):
        footprint_dict = {}
        for ll in ['longitude', 'latitude']:
            footprint_dict[ll] = [
                tile.sel(pol=polarization)[ll].isel(line=a, sample=x).values for a, x in
                [(0, 0), (0, -1), (-1, -1), (-1, 0)]
            ]
        corners = list(zip(footprint_dict['longitude'], footprint_dict['latitude']))
        foot_print = Polygon(corners)

        tiles_footprint.append(foot_print)
    return tiles_footprint


def plot_cartopy_data(ds=None, tiles=None, polarization='VV', file_name='map', tiles_footprint=None):
    """
    Plot SAR dataset or tiles overlaid on a map using Cartopy.

    Parameters:
        ds (xarray.Dataset): SAR dataset.
        tiles (list of xarray.DataArray): List of tiles.
        polarization (str): Polarization to select from dataset or tiles.
        file_name (str): Name of the output plot file.
        tiles_footprint (list of shapely.geometry.Polygon, optional): Footprint of each tile.

    Returns:
        None
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
            ax.pcolormesh(lon.data, lat.data, img.data, transform=ccrs.PlateCarree(), cmap='gray', zorder=1) # , vmin=0, vmax=0.02

        if tiles_footprint is not None:
            for footprint in tiles_footprint:
                patch = mpatches.Polygon(list(zip(*footprint.exterior.xy)), edgecolor='red', linestyle='--',
                                         facecolor='none', transform=ccrs.PlateCarree())
                ax.add_patch(patch)

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)

    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title(file_name)

    plt.show()
