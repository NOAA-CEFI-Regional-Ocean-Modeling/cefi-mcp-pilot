"""Tools for the MCP server; these do the actual work"""

import asyncio
import calendar
from typing import Annotated
from urllib.parse import quote, urlparse

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr
from async_lru import alru_cache
from loguru import logger
from pydantic import Field, HttpUrl

from .utils import find_dim, normalize_lon

Variable = Annotated[
    str,
    Field(
        description='Variable to read from the dataset. '
        'This can be found in the cefi_variable field of the CEFI data catalog metadata'
    ),
]
CEFI_Opendap_URL = Annotated[
    HttpUrl,
    Field(
        description='OpenDAP url to open. This can be found in the '
        'cefi_opendap field of the CEFI data catalog metadata.'
        'This will work best if it is the url for the regridded version of the data.'
    ),
]


def validate_cefi_opendap_url(url: str) -> bool:
    """
    Validate that the OpenDAP URL is from the allowed CEFI domain and path.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is allowed, False otherwise

    Raises:
        ValueError: If the URL is not allowed with explanation
    """
    try:
        parsed = urlparse(url)

        # Check protocol - only allow HTTP or HTTPS
        if parsed.scheme not in ('http', 'https'):
            raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only http/https are allowed.")

        # Check domain - must be exactly psl.noaa.gov
        if parsed.netloc != 'psl.noaa.gov':
            raise ValueError(f"Invalid domain '{parsed.netloc}'. Only 'psl.noaa.gov' is allowed.")

        # Check path - must start with /thredds/dodsC/Projects/CEFI/
        allowed_path_prefix = '/thredds/dodsC/Projects/CEFI/'
        if not parsed.path.startswith(allowed_path_prefix):
            raise ValueError(
                f"Invalid path '{parsed.path}'. Must start with '{allowed_path_prefix}'."
            )

        # Prevent path traversal
        if '..' in parsed.path:
            raise ValueError(f"Invalid path '{parsed.path}'. Path traversal is not allowed.")

        return True

    except Exception as e:
        raise ValueError(f'URL validation failed: {e}') from e


@alru_cache(maxsize=5, ttl=3600)
async def _cached_open_dataset(cefi_opendap_url: str) -> xr.Dataset:
    """Cached dataset opening without timeout to avoid cache key issues."""
    logger.info('Opening dataset (cache miss): {url}', url=cefi_opendap_url)
    return await asyncio.to_thread(xr.open_dataset, cefi_opendap_url)


async def open_dataset(cefi_opendap_url: str, timeout: int = 30) -> xr.Dataset:
    """Get dataset from cache or open it if not cached. Cache expires after 1 hour."""
    # Validate the URL before attempting to open
    validate_cefi_opendap_url(cefi_opendap_url)

    try:
        # Use cached version with timeout wrapper
        return await asyncio.wait_for(_cached_open_dataset(cefi_opendap_url), timeout=timeout)
    except TimeoutError as e:
        raise ValueError(
            f'Dataset access timed out after {timeout} seconds. This dataset may be '
            'very large or the server is slow. Suggestions: 1) Try a monthly dataset, '
            '2) Use regridded data instead of raw data if available, '
            '3) Try a different time period or variable, '
            '4) Retry in a few minutes when server load may be lower.'
        ) from e


def _setup_coord_slice(
    ds: xr.Dataset, variable: str, latitude: float, longitude: float, depth: float
) -> dict[str, float]:
    """Setup coordinate slice for data selection. Raises ValueError on error."""
    try:
        lat_name = find_dim(ds[variable], ['lat', 'yh', 'latitude', 'y', 'yc', 'yq'], 'y')
        lon_name = find_dim(ds[variable], ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    except ValueError as e:
        raise ValueError('Could not determine data lat/lon coordinates') from e

    # Normalize longitude coordinates in dataset and input parameter
    # Note that this mutates the dataset.
    ds[lon_name] = normalize_lon(ds[lon_name])
    normalized_longitude = normalize_lon(longitude)
    coord_slice = {lat_name: latitude, lon_name: normalized_longitude}

    if depth >= 0:
        try:
            depth_name = find_dim(ds[variable], ['z_l', 'zl', 'z_i', 'zi'], 'z')
            coord_slice.update({depth_name: depth})
        except ValueError:
            # This is probably not a critical error, but it should
            # also probably be passed back to the client as a warning.
            logger.warning('Depth was requested but not found. This is ok if it is a 2D dataset')
            # raise ValueError(
            #     'A depth value was requested, but the dataset does not have depth. '
            #     'Do not request a depth if the variable is already at '
            #     'the surface or bottom.'
            # ) from e

    return coord_slice


def _add_forecast_time(ds: xr.Dataset) -> xr.Dataset:
    """Add time dimension to forecast datasets based on lead and init times.
    If lead and init are not present, nothing is done.
    Raises ValueError on error.
    Note, this mutates the dataset.
    """
    if 'lead' in ds.dims and 'time' not in ds.dims and 'init' in ds:
        logger.info('Adding time to forecast')
        if 'init' in ds.dims:
            raise ValueError(
                'This tool cannot yet select from a forecast with multiple initializations.'
            )
        if isinstance(ds.lead.values[0], np.timedelta64):
            valid_time = (ds.init + ds.lead).data
        elif 'units' in ds['lead'].attrs and ds['lead'].attrs['units'] == 'days':
            valid_time = [ds.init.values + pd.Timedelta(days=int(ld)) for ld in ds.lead]
        else:
            valid_time = [ds.init.values + pd.DateOffset(months=int(ld)) for ld in ds.lead]
        ds['time'] = (('lead',), valid_time)
        ds = ds.swap_dims({'lead': 'time'})
    return ds


async def query_variable_metadata(
    cefi_opendap_url: Annotated[
        HttpUrl,
        Field(
            description='OpenDAP url to open. This can be found in the '
            'cefi_opendap field of the CEFI data catalog metadata.'
        ),
    ],
    variable: Variable,
) -> str:
    """Get detailed information for a specific variable by reading it from an opendap URL."""
    try:
        ds = await open_dataset(str(cefi_opendap_url), 30)

        if variable not in ds.data_vars:
            available_vars = list(ds.data_vars.keys())
            return f"Variable '{variable}' not found. Available variables: {available_vars}"

        var_data = ds[variable]

        # Get basic metadata
        metadata = {
            'shape': var_data.shape,
            'dimensions': var_data.dims,
            'units': var_data.attrs.get('units', 'N/A'),
            'long_name': var_data.attrs.get('long_name', 'N/A'),
            'standard_name': var_data.attrs.get('standard_name', 'N/A'),
            'description': var_data.attrs.get('description', 'N/A'),
            'all_attributes': dict(var_data.attrs),
        }

        # Time information if time dimension exists
        if 'time' in var_data.dims and 'time' in ds.coords:
            time_coord = ds.coords['time']
            metadata['temporal_info'] = {
                'time_span': {
                    'start': str(time_coord.min().values),
                    'end': str(time_coord.max().values),
                    'count': len(time_coord),
                },
                'time_units': time_coord.attrs.get('units', 'N/A'),
                'calendar': time_coord.attrs.get('calendar', 'N/A'),
            }

        # Spatial information
        spatial_dims = []
        for dim in var_data.dims:
            if dim in ['lat', 'latitude', 'lon', 'longitude', 'x', 'y'] and dim in ds.coords:
                coord_data = ds.coords[dim]
                spatial_dims.append(
                    {
                        'name': dim,
                        'size': len(coord_data),
                        'range': [
                            float(coord_data.min().values),
                            float(coord_data.max().values),
                        ],
                        'units': coord_data.attrs.get('units', 'N/A'),
                        'long_name': coord_data.attrs.get('long_name', 'N/A'),
                    }
                )

        if spatial_dims:
            metadata['spatial_info'] = spatial_dims

        return f"Metadata for '{variable}':\n{metadata}"

    except ValueError as e:
        return f'URL validation error: {e}'
    except Exception as e:
        return f'Error retrieving metadata: {e!s}'


async def get_variable_point(
    cefi_opendap_url: CEFI_Opendap_URL,
    variable: Variable,
    latitude: Annotated[
        float, Field(description='Latitude of the point to get in degrees.', ge=-90.0, le=90.0)
    ],
    longitude: Annotated[
        float,
        Field(description='Longitude of the point to get in degrees east, ranging from 0 to 360'),
    ],
    year: Annotated[int, Field(description='Year of the single date', ge=1900, le=2100)],
    month: Annotated[int, Field(description='Month of the single date', ge=1, le=12)],
    day: Annotated[
        int,
        Field(
            description='Integer day of the month of the single date. '
            'Only needed if daily data is being read.',
            default=0,  # keep default an int so client will know to use ints
            ge=1,
            le=31,
        ),
    ],
    depth: Annotated[
        float,
        Field(
            description='Depth of the point to get in meters,'
            'with positive values deeper in the ocean. Do not use if'
            '"surface" or "bottom" data are requested.',
            ge=0,
            le=6500,
            default=-1.0,  # keep default a float so client will know to use floats
        ),
    ],
) -> dict:
    """
    Get data for a single variable at a single point in lat and lon.
    For hindcast datasets: returns a single value at the specified time.
    For forecast datasets: returns the complete forecast time series (typically 12 months).
    Use the get_variable_climatology_point tool instead if the long-term average is desired.
    """
    try:
        ds = await open_dataset(str(cefi_opendap_url), 30)
    except ValueError as e:
        return {'Error': f'URL validation error: {e}'}

    # TODO: the variable name could be automatically determined if there is only one
    # TODO: should be sure that the data are reduced to a single value.

    try:
        coord_slice = _setup_coord_slice(ds, variable, latitude, longitude, depth)
        ds = _add_forecast_time(ds)
    except ValueError as e:
        return {'Error': str(e)}

    # Check if this is a forecast dataset (has lead/init dimensions)
    is_forecast = 'lead' in ds.dims or ('init' in ds and 'time' in ds.dims and len(ds.time) <= 366)

    if is_forecast:
        logger.info('Assuming dataset is a forecast')
        # For forecast data, return the complete time series
        try:
            res = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: ds[variable].sel(method='nearest', **coord_slice).squeeze()
                ),
                timeout=30,
            )
        except TimeoutError:
            return {
                'Error': (
                    'Data selection timed out after 30 seconds. This dataset may be '
                    'very large or the operation too complex. Try a different variable or location.'
                )
            }

        # If there are multiple ensemble members, compute the mean
        if 'member' in res.dims:
            std = res.std('member')
            res = res.mean('member')
        else:
            std = None

        # Convert to a list of time-value pairs for easy LLM understanding
        mean_ts = []
        std_ts = []
        for time_val in res.time.values:
            mean_val = float(res.sel(time=time_val).item())
            std_val = 'Unknown' if std is None else float(std.sel(time=time_val).item())
            mean_ts.append(
                {'time': str(pd.Timestamp(time_val).strftime('%Y-%m-%d')), 'value': mean_val}
            )
            std_ts.append(
                {'time': str(pd.Timestamp(time_val).strftime('%Y-%m-%d')), 'value': std_val}
            )

        return {
            'Variable name': ds[variable].attrs.get('long_name', 'Unknown'),
            'Data units': ds[variable].attrs.get('units', 'unknown'),
            'Data type': 'Forecast time series',
            'Forecast initialization': str(ds.init.values) if 'init' in ds else 'N/A',
            'Number of forecast timesteps': len(mean_ts),
            'Best estimate time series': mean_ts,
            'Uncertainty time series (1 standard deviation)': std_ts,
        }

    else:
        logger.info('Assuming dataset is not a forecast')
        # For hindcast data, return single point as before
        # TODO: check what the data frequency is and whether
        # the day should be requested or not.
        if day > 0:
            time_point = f'{year}-{month:02d}-{day:02d}'
        else:
            time_point = f'{year}-{month:02d}'
        try:
            # Wrap data selection in timeout to prevent resource exhaustion
            res = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: (
                        ds[variable]
                        .sel(time=time_point)
                        .sel(method='nearest', **coord_slice)
                        .squeeze()
                    )
                ),
                timeout=30,
            )
        except TimeoutError:
            return {
                'Error': (
                    'Data selection timed out after 30 seconds. This dataset may be '
                    'very large or the operation too complex. Suggestions: 1) Use monthly '
                    'instead of daily data if available, 2) Retry when server load is lower.'
                )
            }
        return {
            'Variable name': ds[variable].attrs.get('long_name', 'Unknown'),
            'Data type': 'Single point',
            'Data value': float(res),
            'Data units': ds[variable].attrs.get('units', 'unknown'),
            'Time requested': time_point,
        }


async def get_variable_climatology_point(
    cefi_opendap_url: CEFI_Opendap_URL,
    variable: Variable,
    latitude: Annotated[
        float, Field(description='Latitude of the point to get in degrees.', ge=-90.0, le=90.0)
    ],
    longitude: Annotated[
        float,
        Field(
            description='Longitude of the point to get in degrees east, ranging from 0 to 360',
            ge=0,
            le=360,
        ),
    ],
    month: Annotated[int, Field(description='Month to find typical value for', ge=1, le=12)],
    depth: Annotated[
        float,
        Field(
            description='Depth of the point to get in meters,'
            'with positive values deeper in the ocean. Do not use if'
            '"surface" or "bottom" data are requested.',
            ge=0,
            le=6500,
            default=-1.0,  # keep default a float so client will know to use floats
        ),
    ],
) -> dict:
    """
    Get the long-term average (also known as the climatology, or the typical value or conditions)
    over a given calendar month for a single variable, at a single point in lat and lon.
    """
    try:
        ds = await open_dataset(str(cefi_opendap_url), 30)
    except ValueError as e:
        return {'Error': f'URL validation error: {e}'}

    # TODO: the variable name could be automatically determined if there is only one
    # TODO: should be sure that the data are reduced to a single value.
    if 'init' in ds.dims or 'lead' in ds.dims:
        return {
            'Error': (
                'This tool cannot get climatology from a forecast. Instead, '
                'use the same variable with "_anom" added to the end of the variable name '
                'to determine if the forecast is above or below normal, or use the '
                'get_variable_climatology_point tool with the hindcast simulation '
                'to get the actual average value.'
            )
        }

    try:
        coord_slice = _setup_coord_slice(ds, variable, latitude, longitude, depth)
    except ValueError as e:
        return {'Error': str(e)}
    try:
        # Wrap climatology calculation in timeout to prevent resource exhaustion
        logger.info('Beginning to calculate climatology')
        res = await asyncio.wait_for(
            asyncio.to_thread(
                lambda: (
                    ds[variable]
                    .sel(time=ds['time.month'] == month)
                    .sel(method='nearest', **coord_slice)
                    .mean('time')
                    .squeeze()
                )
            ),
            timeout=30,
        )
    except TimeoutError:
        return {
            'Error': (
                'Climatology calculation timed out after 30 seconds. '
                'Suggestions: 1) Use monthly instead of daily data if available, '
                '2) Try a different variable or location, 3) Retry when server load is lower.'
            )
        }
    return {
        'Variable name': res.attrs.get('long_name', 'Unknown'),
        f'Average for {calendar.month_name[month]}': float(res),
        'Data units': res.attrs.get('units', 'unknown'),
    }


async def geocode_ocean_place(
    place_name: Annotated[
        str,
        Field(
            description=(
                'Name of the ocean place to geocode (e.g., "Gulf of Maine", "Chesapeake Bay")'
            )
        ),
    ],
) -> dict:
    """
    Geocode ocean place names using the Marine Regions Gazetteer API.

    Returns geographic information including coordinates, names, and boundaries
    for ocean places, water bodies, and marine regions.
    """
    try:
        # URL encode the place name and construct the API URL
        encoded_place = quote(place_name)
        url = (
            f'https://www.marineregions.org/rest/getGazetteerRecordsByName.json/'
            f'{encoded_place}/?like=true&fuzzy=true&offset=0&count=10'
        )

        logger.info('Geocoding ocean place: {place}', place=place_name)

        timeout = aiohttp.ClientTimeout(total=30.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

            if not data:
                return {
                    'Place name': place_name,
                    'Results found': 0,
                    'Message': (
                        f'No results found for "{place_name}". Try a different name or spelling.'
                    ),
                }

            results = []
            for item in data:
                result = {
                    'Place ID': item.get('mrgid'),
                    'Preferred name': item.get('preferredGazetteerName'),
                    'Place type': item.get('placeType'),
                    'Latitude': item.get('latitude'),
                    'Longitude': item.get('longitude'),
                    'Minimum latitude': item.get('minLatitude'),
                    'Maximum latitude': item.get('maxLatitude'),
                    'Minimum longitude': item.get('minLongitude'),
                    'Maximum longitude': item.get('maxLongitude'),
                }
                # Only include non-null values
                result = {k: v for k, v in result.items() if v is not None}
                results.append(result)

            return {
                'Place name': place_name,
                'Results found': len(results),
                'Results': results,
            }

    except aiohttp.ServerTimeoutError:
        return {'Error': 'Request timed out after 30 seconds. The Marine Regions API may be slow.'}
    except aiohttp.ClientResponseError as e:
        return {
            'Error': (
                f'HTTP error {e.status}: {e.message}. '
                'This may mean that the geocoding API could not identify the place name. '
                'Try a shorter or simpler place name.'
            )
        }
    except Exception as e:
        logger.error('Error geocoding place {place}: {error}', place=place_name, error=str(e))
        return {'Error': f'Geocoding failed: {e!s}'}
