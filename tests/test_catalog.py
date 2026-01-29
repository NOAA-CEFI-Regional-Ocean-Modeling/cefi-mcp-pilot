import json
import re
from datetime import UTC

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


@pytest.mark.asyncio
async def test_computed_fields_in_json_schema():
    """Test that computed fields are included in the JSON schema.

    This ensures FastMCP can generate a valid output schema that includes
    the computed datetime fields. Without this configuration, the serialized
    data would include fields not defined in the schema, causing validation errors.
    """
    schema = CEFIDatasetMetadata.model_json_schema()

    # Verify computed fields are in the schema
    assert 'start_date' in schema['properties'], 'start_date must be in JSON schema'
    assert 'end_date' in schema['properties'], 'end_date must be in JSON schema'
    assert 'release_date' in schema['properties'], 'release_date must be in JSON schema'
    assert 'init_date' in schema['properties'], 'init_date must be in JSON schema'

    # Verify release_date has the correct format
    release_schema = schema['properties']['release_date']
    assert release_schema['type'] == 'string', 'release_date should be string type in schema'
    assert release_schema['format'] == 'date-time', 'release_date should have date-time format'
    assert release_schema.get('readOnly') is True, 'computed fields should be readOnly'


@pytest.mark.asyncio
async def test_datetime_fields_are_timezone_aware(sample_catalog_data):
    """Test that computed datetime fields are timezone-aware and serialize to RFC 3339 format.

    This test ensures datetime fields include UTC timezone information, which is required
    for RFC 3339 compliant "date-time" format used by JSON Schema validation.
    Without timezone info, serialization produces timestamps like "2023-05-20T00:00:00"
    which fail validation. With UTC timezone, they serialize to "2023-05-20T00:00:00Z".
    """
    dataset = CEFIDatasetMetadata(**sample_catalog_data['data1'])

    # Test 1: Verify datetime objects have timezone info (are timezone-aware)
    assert dataset.release_date.tzinfo is not None, 'release_date should be timezone-aware'
    assert dataset.release_date.tzinfo == UTC, 'release_date should use UTC timezone'

    # start_date and end_date can be str or datetime, so check if datetime
    if isinstance(dataset.start_date, type(dataset.release_date)):
        assert dataset.start_date.tzinfo is not None, 'start_date should be timezone-aware'
        assert dataset.start_date.tzinfo == UTC, 'start_date should use UTC timezone'

    if isinstance(dataset.end_date, type(dataset.release_date)):
        assert dataset.end_date.tzinfo is not None, 'end_date should be timezone-aware'
        assert dataset.end_date.tzinfo == UTC, 'end_date should use UTC timezone'

    if isinstance(dataset.init_date, type(dataset.release_date)):
        assert dataset.init_date.tzinfo is not None, 'init_date should be timezone-aware'
        assert dataset.init_date.tzinfo == UTC, 'init_date should use UTC timezone'

    # Test 2: Verify JSON serialization produces RFC 3339 format with 'Z' suffix
    json_str = dataset.model_dump_json()
    data = json.loads(json_str)

    # Check that datetime strings end with 'Z' (UTC timezone indicator)
    assert data['release_date'].endswith('Z'), 'release_date must serialize with Z suffix'

    if isinstance(dataset.start_date, type(dataset.release_date)):
        assert data['start_date'].endswith('Z'), 'start_date must serialize with Z suffix'

    if isinstance(dataset.end_date, type(dataset.release_date)):
        assert data['end_date'].endswith('Z'), 'end_date must serialize with Z suffix'

    if isinstance(dataset.init_date, type(dataset.release_date)):
        assert data['init_date'].endswith('Z'), 'init_date must serialize with Z suffix'

    # Test 3: Verify the format matches RFC 3339 date-time pattern
    # Format should be: YYYY-MM-DDTHH:MM:SSZ
    rfc3339_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'
    assert re.match(rfc3339_pattern, data['release_date']), (
        f"release_date '{data['release_date']}' must match RFC 3339 format"
    )

    if isinstance(dataset.start_date, type(dataset.release_date)):
        assert re.match(rfc3339_pattern, data['start_date']), (
            f"start_date '{data['start_date']}' must match RFC 3339 format"
        )

    if isinstance(dataset.end_date, type(dataset.release_date)):
        assert re.match(rfc3339_pattern, data['end_date']), (
            f"end_date '{data['end_date']}' must match RFC 3339 format"
        )


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
async def test_list_variables_with_search_exact_match(sample_catalog):
    """Test listing variables with exact substring match"""
    # Search for 'ion' should find btm_co3_ion
    variables = sample_catalog.list_variables(search_term='ion')
    assert 'btm_co3_ion' in variables
    assert len(variables) == 1


@pytest.mark.asyncio
async def test_list_variables_with_search_case_insensitive(sample_catalog):
    """Test that search is case-insensitive"""
    # Search for 'ION' should find btm_co3_ion
    variables = sample_catalog.list_variables(search_term='ION')
    assert 'btm_co3_ion' in variables
    assert len(variables) == 1


@pytest.mark.asyncio
async def test_list_variables_with_search_in_description(sample_catalog):
    """Test searching in variable descriptions"""
    # Search for 'Carbonate' should find variables with 'Carbonate' in description
    variables = sample_catalog.list_variables(search_term='Carbonate')
    assert 'btm_co3_ion' in variables
    assert len(variables) >= 1


@pytest.mark.asyncio
async def test_list_variables_with_search_multiple_matches(sample_catalog):
    """Test searching returns multiple matches when appropriate"""
    # Search for 'btm' should find all variables starting with 'btm'
    variables = sample_catalog.list_variables(search_term='btm')
    assert len(variables) == 4


@pytest.mark.asyncio
async def test_list_variables_with_search_no_matches(sample_catalog):
    """Test searching with no matches returns empty dict"""
    variables = sample_catalog.list_variables(search_term='nonexistent_variable_xyz')
    assert len(variables) == 0
    assert isinstance(variables, dict)


@pytest.mark.asyncio
async def test_list_variables_with_search_fuzzy_match(sample_catalog):
    """Test fuzzy matching finds similar terms"""
    # Search for 'solubility' should fuzzy match to 'Solubility' in descriptions
    variables = sample_catalog.list_variables(search_term='solubility')
    assert len(variables) >= 1


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
