import numpy as np
import pytest
import xarray as xr

from cefi_mcp.utils import find_dim


@pytest.mark.parametrize('dim_name', ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'])
def test_find_x_dim_single_valid_dimension(dim_name):
    """Test find_dim with each supported x-dimension name."""
    # Create a simple DataArray with the specified dimension
    data = xr.DataArray(np.random.rand(10), dims=[dim_name], coords={dim_name: np.arange(10)})

    result = find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    assert result == dim_name


@pytest.mark.parametrize('dim_name', ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'])
def test_find_x_dim_with_dataset(dim_name):
    """Test find_dim with Dataset objects."""
    # Create a Dataset with the specified dimension
    data = xr.Dataset(
        {'temperature': ([dim_name, 'time'], np.random.rand(10, 5))},
        coords={dim_name: np.arange(10), 'time': np.arange(5)},
    )

    result = find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    assert result == dim_name


def test_find_x_dim_multiple_valid_dimensions():
    """Test find_dim returns first match when multiple valid dimensions exist."""
    # Create DataArray with multiple valid x-dimension names
    # The function returns the first dimension found in ds.dims that matches the search list
    data = xr.DataArray(
        np.random.rand(10, 8, 6),
        dims=['longitude', 'lon', 'xh'],  # longitude appears first in dims
        coords={'longitude': np.arange(10), 'lon': np.arange(8), 'xh': np.arange(6)},
    )

    result = find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    assert result == 'longitude'  # First in ds.dims that matches


def test_find_x_dim_with_y_dimension():
    """Test find_dim works correctly when y-dimensions are also present."""
    data = xr.DataArray(
        np.random.rand(10, 8),
        dims=['lon', 'lat'],
        coords={'lon': np.arange(10), 'lat': np.arange(8)},
    )

    result = find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    assert result == 'lon'


def test_find_x_dim_no_matching_dimension():
    """Test find_dim raises ValueError when no matching dimension found."""
    data = xr.DataArray(
        np.random.rand(10, 8),
        dims=['time', 'depth'],
        coords={'time': np.arange(10), 'depth': np.arange(8)},
    )

    with pytest.raises(ValueError):
        find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')


def test_find_x_dim_empty_dataset():
    """Test find_dim with empty dataset."""
    data = xr.Dataset()

    with pytest.raises(ValueError):
        find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')


def test_find_x_dim_no_dimensions():
    """Test find_dim with scalar DataArray (no dimensions)."""
    data = xr.DataArray(42.0)

    with pytest.raises(ValueError):
        find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')


def test_find_all_three_dims_3d_oceanographic_data():
    """Test find_dim function with realistic 3D oceanographic data structure."""
    # Create a realistic 3D oceanographic dataset
    data = xr.Dataset(
        {
            'temperature': (['time', 'z_l', 'lat', 'lon'], np.random.rand(12, 50, 180, 360)),
            'salinity': (['time', 'z_l', 'lat', 'lon'], np.random.rand(12, 50, 180, 360)),
        },
        coords={
            'time': np.arange(12),
            'z_l': np.linspace(0, 5000, 50),
            'lat': np.linspace(-90, 90, 180),
            'lon': np.linspace(-180, 180, 360),
        },
    )

    x_dim = find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    y_dim = find_dim(data, ['lat', 'yh', 'latitude', 'y', 'yc', 'yq'], 'y')
    z_dim = find_dim(data, ['z_l', 'zl', 'z_i', 'zi'], 'z')

    assert x_dim == 'lon'
    assert y_dim == 'lat'
    assert z_dim == 'z_l'


def test_find_all_three_dims_mom6_style_3d():
    """Test find_dim function with MOM6-style 3D grid coordinates."""
    # Create MOM6-style dataset with xh/yh/z_l dimensions
    data = xr.Dataset(
        {
            'temp': (['time', 'z_l', 'yh', 'xh'], np.random.rand(12, 75, 100, 200)),
            'salt': (['time', 'z_l', 'yh', 'xh'], np.random.rand(12, 75, 100, 200)),
            'u_vel': (['time', 'zi', 'yh', 'xq'], np.random.rand(12, 76, 100, 201)),
        },
        coords={
            'time': np.arange(12),
            'z_l': np.linspace(0, 5000, 75),
            'zi': np.linspace(0, 5000, 76),
            'yh': np.linspace(20, 60, 100),
            'xh': np.linspace(-80, -40, 200),
            'xq': np.linspace(-80, -40, 201),
        },
    )

    x_dim = find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    y_dim = find_dim(data, ['lat', 'yh', 'latitude', 'y', 'yc', 'yq'], 'y')
    z_dim = find_dim(data, ['z_l', 'zl', 'z_i', 'zi'], 'z')

    assert x_dim == 'xh'  # First x-dimension found
    assert y_dim == 'yh'
    assert z_dim == 'z_l'  # First z-dimension found


def test_find_all_three_dims_mixed_conventions():
    """Test find_dim function with mixed coordinate naming conventions."""
    # Test with longitude, latitude, and zi (mixed conventions)
    data = xr.Dataset(
        {
            'ocean_data': (
                ['time', 'zi', 'latitude', 'longitude'],
                np.random.rand(12, 30, 80, 120),
            ),
        },
        coords={
            'time': np.arange(12),
            'zi': np.linspace(0, 1000, 30),
            'latitude': np.linspace(25, 45, 80),
            'longitude': np.linspace(-75, -65, 120),
        },
    )

    x_dim = find_dim(data, ['lon', 'xh', 'longitude', 'x', 'xc', 'xq'], 'x')
    y_dim = find_dim(data, ['lat', 'yh', 'latitude', 'y', 'yc', 'yq'], 'y')
    z_dim = find_dim(data, ['z_l', 'zl', 'z_i', 'zi'], 'z')

    assert x_dim == 'longitude'
    assert y_dim == 'latitude'
    assert z_dim == 'zi'
