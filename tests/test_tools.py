from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from cefi_mcp.tools import (
    _analyze_temporal_info,
    _cached_open_dataset,
    _validate_date,
    open_dataset,
    query_variable_metadata,
)


@pytest.fixture
def sample_dataset():
    """Create a realistic xarray Dataset for testing"""
    # Create realistic coordinate arrays
    time = np.arange(
        np.datetime64('2023-01-01'), np.datetime64('2023-01-11'), dtype='datetime64[D]'
    )
    lat = np.linspace(-90, 90, 180)
    lon = np.linspace(-180, 180, 360)

    # Create realistic data arrays
    temperature_data = 15 + 10 * np.random.random((10, 180, 360))
    salinity_data = 35 + 2 * np.random.random((10, 180, 360))

    # Create Dataset with realistic structure
    ds = xr.Dataset(
        {
            'temperature': (
                ('time', 'lat', 'lon'),
                temperature_data,
                {
                    'units': 'degrees_C',
                    'long_name': 'Temperature',
                    'standard_name': 'sea_water_temperature',
                },
            ),
            'salinity': (
                ('time', 'lat', 'lon'),
                salinity_data,
                {
                    'units': 'psu',
                    'long_name': 'Salinity',
                    'standard_name': 'sea_water_salinity',
                },
            ),
        },
        coords={
            'time': ('time', time, {'units': 'days since 1900-01-01', 'calendar': 'gregorian'}),
            'lat': ('lat', lat, {'units': 'degrees_north', 'long_name': 'Latitude'}),
            'lon': ('lon', lon, {'units': 'degrees_east', 'long_name': 'Longitude'}),
        },
        attrs={'title': 'Test Dataset', 'source': 'Test'},
    )

    return ds


# Tests for open_dataset function


@pytest.mark.asyncio
@patch('cefi_mcp.tools.xr.open_dataset')
async def test_open_dataset_success(mock_xr_open, sample_dataset):
    """Test successful dataset opening"""
    mock_xr_open.return_value = sample_dataset

    # Clear cache first
    _cached_open_dataset.cache_clear()

    result = await open_dataset('http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc')
    assert result is sample_dataset
    assert 'temperature' in result.data_vars
    assert 'salinity' in result.data_vars
    mock_xr_open.assert_called_once()


@pytest.mark.asyncio
@patch('cefi_mcp.tools.xr.open_dataset')
async def test_open_dataset_caching(mock_xr_open, sample_dataset):
    """Test that datasets are cached properly"""
    mock_xr_open.return_value = sample_dataset

    # Clear cache first
    _cached_open_dataset.cache_clear()

    # First call
    result1 = await open_dataset('http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc')
    # Second call with same URL should return cached result
    result2 = await open_dataset('http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc')

    assert result1 is result2  # Should be the exact same object from cache
    assert result1.attrs['title'] == 'Test Dataset'
    # xr.open_dataset should only be called once due to caching
    mock_xr_open.assert_called_once()


# Tests for query_variable_metadata function


@pytest.mark.asyncio
@patch('cefi_mcp.tools.open_dataset')
async def test_query_variable_metadata_success(mock_open, sample_dataset):
    """Test successful variable metadata query with real data"""
    mock_open.return_value = sample_dataset

    result = await query_variable_metadata(
        'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc', 'temperature'
    )

    # Verify metadata content is properly extracted
    assert "Metadata for 'temperature'" in result
    assert 'degrees_C' in result
    assert 'Temperature' in result
    assert 'sea_water_temperature' in result
    assert "'shape': (10, 180, 360)" in result
    assert "'dimensions': ('time', 'lat', 'lon')" in result
    mock_open.assert_called_once()


@pytest.mark.asyncio
@patch('cefi_mcp.tools.open_dataset')
async def test_query_variable_metadata_variable_not_found(mock_open, sample_dataset):
    """Test querying non-existent variable shows available options"""
    mock_open.return_value = sample_dataset

    result = await query_variable_metadata(
        'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc', 'nonexistent_var'
    )

    assert "Variable 'nonexistent_var' not found" in result
    assert 'Available variables:' in result
    assert 'temperature' in result
    assert 'salinity' in result


@pytest.mark.asyncio
@patch('cefi_mcp.tools.open_dataset')
async def test_query_variable_metadata_connection_error(mock_open):
    """Test error handling when dataset cannot be opened"""
    mock_open.side_effect = Exception('Connection failed')

    result = await query_variable_metadata(
        'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc', 'temperature'
    )

    assert 'Error retrieving metadata' in result
    assert 'Connection failed' in result


@pytest.mark.asyncio
@patch('cefi_mcp.tools.xr.open_dataset')
async def test_open_dataset_network_timeout(mock_xr_open):
    """Test handling of network timeouts"""
    mock_xr_open.side_effect = TimeoutError('Request timed out')

    # Clear cache first
    _cached_open_dataset.cache_clear()

    # The open_dataset function converts TimeoutError to ValueError with helpful message
    with pytest.raises(ValueError, match='Dataset access timed out'):
        await open_dataset('http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc')


@pytest.mark.asyncio
@patch('cefi_mcp.tools.xr.open_dataset')
async def test_open_dataset_invalid_file_format(mock_xr_open):
    """Test handling of invalid NetCDF file formats"""
    mock_xr_open.side_effect = ValueError('Not a valid NetCDF file')

    # Clear cache first
    _cached_open_dataset.cache_clear()

    with pytest.raises(ValueError, match='Not a valid NetCDF file'):
        await open_dataset('http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc')


# Tests for _validate_date helper function


def test_validate_date_valid_full_date():
    """Test validation of a valid full date"""
    # Should not raise
    _validate_date(2023, 1, 15)
    _validate_date(2023, 12, 31)
    _validate_date(2020, 2, 29)  # Leap year


def test_validate_date_valid_monthly():
    """Test validation with day=0 (monthly data)"""
    # Should not raise
    _validate_date(2023, 1, 0)
    _validate_date(2023, 12, 0)


def test_validate_date_invalid_day_june_31():
    """Test that June 31st is rejected"""
    with pytest.raises(ValueError, match=r'Invalid date.*June has 30 days'):
        _validate_date(2023, 6, 31)


def test_validate_date_invalid_day_february_leap_year():
    """Test that February 30th is invalid even in leap years"""
    with pytest.raises(ValueError, match=r'Invalid date.*February has 29 days'):
        _validate_date(2020, 2, 30)


def test_validate_date_february_29_non_leap_year():
    """Test that February 29th is invalid in non-leap years"""
    with pytest.raises(ValueError, match=r'Invalid date.*February has 28 days'):
        _validate_date(2023, 2, 29)


def test_validate_date_april_31():
    """Test that April 31st is rejected"""
    with pytest.raises(ValueError, match=r'Invalid date.*April has 30 days'):
        _validate_date(2023, 4, 31)


# Tests for _analyze_temporal_info helper function


@pytest.fixture
def daily_dataset():
    """Create a dataset with daily frequency"""
    time = np.arange(
        np.datetime64('2023-01-01'), np.datetime64('2023-01-11'), dtype='datetime64[D]'
    )
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    temp_data = 15 + 10 * np.random.random((10, 10, 10))

    ds = xr.Dataset(
        {
            'temperature': (
                ('time', 'lat', 'lon'),
                temp_data,
                {'units': 'degrees_C', 'long_name': 'Temperature'},
            ),
        },
        coords={
            'time': ('time', time, {'units': 'days since 1900-01-01'}),
            'lat': ('lat', lat),
            'lon': ('lon', lon),
        },
    )
    return ds


@pytest.fixture
def monthly_dataset():
    """Create a dataset with monthly frequency"""
    time = np.arange(
        np.datetime64('2023-01-01'), np.datetime64('2023-12-01'), dtype='datetime64[M]'
    )
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    temp_data = 15 + 10 * np.random.random((len(time), 10, 10))

    ds = xr.Dataset(
        {
            'temperature': (
                ('time', 'lat', 'lon'),
                temp_data,
                {'units': 'degrees_C', 'long_name': 'Temperature'},
            ),
        },
        coords={
            'time': ('time', time, {'units': 'days since 1900-01-01'}),
            'lat': ('lat', lat),
            'lon': ('lon', lon),
        },
    )
    return ds


@pytest.fixture
def no_time_dimension_dataset():
    """Create a dataset without time dimension"""
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    temp_data = 15 + 10 * np.random.random((10, 10))

    ds = xr.Dataset(
        {
            'temperature': (
                ('lat', 'lon'),
                temp_data,
                {'units': 'degrees_C', 'long_name': 'Temperature'},
            ),
        },
        coords={
            'lat': ('lat', lat),
            'lon': ('lon', lon),
        },
    )
    return ds


@pytest.fixture
def single_timestep_dataset():
    """Create a dataset with single timestep"""
    time = np.array([np.datetime64('2023-01-01')])
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 10)

    temp_data = 15 + 10 * np.random.random((1, 10, 10))

    ds = xr.Dataset(
        {
            'temperature': (
                ('time', 'lat', 'lon'),
                temp_data,
                {'units': 'degrees_C', 'long_name': 'Temperature'},
            ),
        },
        coords={
            'time': ('time', time),
            'lat': ('lat', lat),
            'lon': ('lon', lon),
        },
    )
    return ds


def test_analyze_temporal_info_daily(daily_dataset):
    """Test temporal analysis with daily frequency"""
    result = _analyze_temporal_info(daily_dataset, 'temperature')

    assert result is not None
    assert result['frequency'] == 'Daily'
    assert '2023-01-01' in result['start']
    assert '2023-01-10' in result['end']
    assert result['count'] == 10
    assert len(result['sample_dates']) == 5  # Should sample 5 dates when > 5 timesteps


def test_analyze_temporal_info_monthly(monthly_dataset):
    """Test temporal analysis with monthly frequency"""
    result = _analyze_temporal_info(monthly_dataset, 'temperature')

    assert result is not None
    assert 'Monthly' in result['frequency']  # Could be 'Monthly', 'Monthly (start)', etc.
    assert '2023-01' in result['start']
    assert '2023-11' in result['end']
    assert result['count'] == 11
    assert len(result['sample_dates']) == 5  # Should sample 5 dates when > 5 timesteps


def test_analyze_temporal_info_sample_dates_downsampling(monthly_dataset):
    """Test that sample dates are properly downsampled when > 5 timesteps"""
    result = _analyze_temporal_info(monthly_dataset, 'temperature')

    assert len(result['sample_dates']) == 5
    # First and last samples should be the start and end
    assert '2023-01' in result['sample_dates'][0]
    assert '2023-11' in result['sample_dates'][-1]


def test_analyze_temporal_info_no_time_dimension(no_time_dimension_dataset):
    """Test that None is returned when variable has no time dimension"""
    result = _analyze_temporal_info(no_time_dimension_dataset, 'temperature')

    assert result is None


def test_analyze_temporal_info_single_timestep(single_timestep_dataset):
    """Test temporal analysis with single timestep"""
    result = _analyze_temporal_info(single_timestep_dataset, 'temperature')

    assert result is not None
    assert result['count'] == 1
    assert len(result['sample_dates']) == 1
    assert '2023-01-01' in result['sample_dates'][0]
