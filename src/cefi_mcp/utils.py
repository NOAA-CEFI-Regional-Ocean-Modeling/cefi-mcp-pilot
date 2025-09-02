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
