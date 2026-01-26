import pytest
import pytest_asyncio

from cefi_mcp.catalog import CEFIDatasetMetadata, load_catalog_file


@pytest_asyncio.fixture
async def sample_catalog_data():
    """Load real catalog data from sample file for testing"""
    catalog = await load_catalog_file('tests/sample_cat.json')
    # Return raw data dict for tests that need it
    return {
        dataset_id: dataset_metadata.model_dump()
        for dataset_id, dataset_metadata in catalog.datasets.items()
    }


@pytest_asyncio.fixture
async def sample_catalog():
    """Load real catalog from sample file for testing"""
    return await load_catalog_file('tests/sample_cat.json')


# Tests for CEFIDatasetMetadata model


@pytest.mark.asyncio
async def test_model_creation(sample_catalog_data):
    """Test creating a dataset metadata model"""
    dataset = CEFIDatasetMetadata(**sample_catalog_data['data1'])
    assert dataset.cefi_variable == 'btm_co3_ion'
    assert dataset.cefi_long_name == 'Bottom Carbonate Ion'
    assert dataset.cefi_output_frequency == 'daily'
    assert (
        str(dataset.cefi_opendap)
        == 'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northwest_atlantic/full_domain/hindcast/daily/raw/r20230520/btm_co3_ion.nwa.full.hcast.daily.raw.r20230520.199301-201912.nc'
    )


@pytest.mark.asyncio
async def test_computed_fields(sample_catalog_data):
    """Test computed fields work correctly"""
    dataset = CEFIDatasetMetadata(**sample_catalog_data['data1'])
    assert dataset.release_date.year == 2023
    assert dataset.release_date.month == 5
    assert dataset.release_date.day == 20


# Tests for CEFIDataCatalog functionality


@pytest.mark.asyncio
async def test_get_dataset_from_catalog(sample_catalog):
    """Test retrieving a specific dataset"""
    dataset = sample_catalog.get_dataset_from_catalog('data1')
    assert dataset is not None
    assert dataset.cefi_variable == 'btm_co3_ion'

    # Test non-existent dataset
    dataset = sample_catalog.get_dataset_from_catalog('nonexistent')
    assert dataset is None


@pytest.mark.asyncio
async def test_list_variables(sample_catalog):
    """Test listing all variables"""
    variables = sample_catalog.list_variables()
    assert 'btm_co3_ion' in variables
    assert 'btm_co3_sol_arag' in variables
    assert 'btm_co3_sol_calc' in variables
    assert 'btm_htotal' in variables
    assert len(variables) == 4


@pytest.mark.asyncio
async def test_filter_by_variable(sample_catalog):
    """Test filtering by variable name"""
    filtered = sample_catalog.filter_by_variable('btm_co3_ion')
    assert len(filtered) == 1
    assert 'data1' in filtered

    # Test non-existent variable
    filtered = sample_catalog.filter_by_variable('nonexistent')
    assert len(filtered) == 0


@pytest.mark.asyncio
async def test_filter_by_frequency(sample_catalog):
    """Test filtering by output frequency"""
    daily = sample_catalog.filter_by_frequency('daily')
    # All 4 datasets in sample_cat.json have 'daily' frequency
    assert len(daily) == 4
    for i in range(1, 5):
        assert f'data{i}' in daily
    # Test non-existent frequency
    monthly = sample_catalog.filter_by_frequency('monthly')
    assert len(monthly) == 0


# Tests for catalog file loading


@pytest.mark.asyncio
async def test_load_catalog_file():
    """Test loading real catalog from sample file"""
    catalog = await load_catalog_file('tests/sample_cat.json')
    assert len(catalog.datasets) == 4
    for i in range(1, 5):
        assert f'data{i}' in catalog.datasets

    variables = catalog.list_variables()
    expected = {
        'btm_co3_ion': 'Bottom Carbonate Ion',
        'btm_co3_sol_arag': 'Bottom Aragonite Solubility',
        'btm_co3_sol_calc': 'Bottom Calcite Solubility',
        'btm_htotal': 'Bottom Htotal',
    }
    assert variables == expected
    assert 'tos' not in variables


@pytest.mark.asyncio
async def test_load_nonexistent_file():
    """Test loading non-existent file raises error"""
    with pytest.raises(FileNotFoundError):
        await load_catalog_file('nonexistent.json')


@pytest.mark.asyncio
async def test_load_catalog_file_path_traversal():
    """Test that path traversal attempts are blocked"""
    with pytest.raises(ValueError, match=r'Access denied.*forbidden'):
        await load_catalog_file('../../../etc/passwd')

    with pytest.raises(ValueError, match=r'Access denied.*forbidden'):
        await load_catalog_file('/etc/passwd')


@pytest.mark.asyncio
async def test_load_catalog_file_forbidden_patterns():
    """Test that paths with forbidden patterns are blocked"""
    with pytest.raises(ValueError, match=r'Access denied.*forbidden'):
        await load_catalog_file('../config.json')

    with pytest.raises(ValueError, match=r'Access denied.*forbidden'):
        await load_catalog_file('/home/user/file.json')


@pytest.mark.asyncio
async def test_load_catalog_file_outside_project_directory():
    """Test that files outside project directory are blocked"""
    with pytest.raises(ValueError, match=r'Access denied.*forbidden'):
        await load_catalog_file('/tmp/malicious.json')
