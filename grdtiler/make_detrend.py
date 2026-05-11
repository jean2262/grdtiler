import logging
import yaml

import numpy as np
import xarray as xr
import xsar
import xsarsea
from pathlib import Path

logger = logging.getLogger(__name__)


def _detect_mission(filename: str) -> str:
    name = filename.upper()
    if name.startswith("S1"):
        return "S1"
    elif name.startswith("RS2"):
        return "RS2"
    elif name.startswith("RCM"):
        return "RCM"
    raise ValueError(f"Unsupported mission in filename: {filename}. Expected S1, RS2 or RCM.")


def _load_dataset(path: str | xr.Dataset, resolution: str = None) -> xr.Dataset:
    if isinstance(path, xr.Dataset):
        return path

    path_obj = Path(path)
    if path_obj.suffix.lower() == ".nc":
        return xr.open_dataset(path_obj)

    SUPPORTED_PREFIXES = ("S1", "RS2", "RCM")
    if path_obj.name.startswith(SUPPORTED_PREFIXES):
        return xsar.open_dataset(str(path_obj), resolution=resolution)

    raise ValueError(
        f"Unsupported file or dataset type: {path}. "
        f"Expected a .nc file or a SAR product starting with {SUPPORTED_PREFIXES}."
    )


def _move_valid_line_to_zero(inc_angle: xr.DataArray) -> xr.DataArray:
    nan_mask = np.isnan(inc_angle.values)
    nan_counts = nan_mask.sum(axis=tuple(range(1, nan_mask.ndim)))
    valid_lines = np.where(nan_counts == 0)[0]
    first_valid_line = valid_lines[0] if valid_lines.size > 0 else np.argmin(nan_counts)
    sliced = inc_angle.isel(line=slice(first_valid_line, None))
    return sliced.assign_coords(line=np.arange(sliced.sizes["line"]))


def make_detrend(path: str | xr.Dataset, config: str | dict, resolution: str = None) -> xr.Dataset:
    """
    Load a SAR dataset and compute sigma0_detrend for all polarisations.

    Args:
        path (str or xr.Dataset): Path to the SAR product (.SAFE or .nc) or a
            pre-loaded xarray Dataset.
        config (str or dict): Path to the GMF model configuration YAML, or a
            pre-loaded configuration dict (avoids repeated file I/O).
        resolution (str, optional): Target resolution forwarded to
            ``xsar.open_dataset`` (e.g. ``"400m"``). Defaults to None.

    Returns:
        xr.Dataset: The dataset with ``sigma0_no_nan`` and ``sigma0_detrend``
        variables added.
    """
    dataset = _load_dataset(path, resolution=resolution)

    if isinstance(config, str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)

    filename = dataset.attrs.get("safe") or dataset.attrs.get("name", "")
    if ":" in filename:
        filename = Path(filename.split(":")[1]).name

    mission = _detect_mission(filename)

    dataset["sigma0_no_nan"] = xr.where(
        dataset["land_mask"], np.nan, dataset["sigma0"]
    )

    pols = list(dataset.pol.values)
    if any(p in ("HH", "HV") for p in pols):
        xsarsea.windspeed.register_nc_luts(config["gmf_base_path"])

    sigma0_detrends = []
    for pol in pols:
        gmf_model = config["gmf_models"][mission][f"GMF_{pol}_NAME"]
        inc_angle_cleaned = _move_valid_line_to_zero(dataset.sel(pol=pol).incidence)
        sigma0_detrends.append(
            xsarsea.sigma0_detrend(
                sigma0=dataset.sel(pol=pol).sigma0,
                inc_angle=inc_angle_cleaned,
                model=gmf_model,
            )
        )

    dataset["sigma0_detrend"] = xr.concat(sigma0_detrends, dim="pol")

    return dataset
