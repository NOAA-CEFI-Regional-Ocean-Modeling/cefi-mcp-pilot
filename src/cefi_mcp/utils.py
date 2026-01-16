import numpy as np
import xarray as xr


def find_dim(ds: xr.Dataset | xr.DataArray, possible_names: list[str], dim_type: str) -> str:
    """Find a dimension by searching for possible names."""
    try:
        name = next(d for d in ds.dims if d in possible_names)
    except StopIteration as err:
        raise ValueError(f'Could not find a {dim_type} dimension') from err
    else:
        return str(name)


def normalize_lon(lon):
    """
    Normalize longitude(s) to the range (0, 360].
    """
    return lon % 360


def format_float(value: float, context: str = 'data') -> str:
    """
    Format floating point numbers for LLM-friendly display.
    """

    # Handle edge cases
    if np.isnan(value):
        return 'NaN'
    if np.isinf(value):
        return 'inf' if value > 0 else '-inf'
    if value == 0.0:
        return '0'

    sig_figs = 4 if context == 'coordinate' else 5
    return f'{value:.{sig_figs}g}'
