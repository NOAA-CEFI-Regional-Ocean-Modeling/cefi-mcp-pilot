"""Pydantic models for CEFI NetCDF metadata"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import aiofiles
import aiohttp
from async_lru import alru_cache
from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, NaiveDatetime, computed_field

from .tools import Variable

Region = Annotated[
    Literal['northwest_atlantic', 'northeast_pacific'], Field(description='CEFI model domain')
]
Simulation = Annotated[
    Literal['hindcast', 'seasonal_reforecast', 'seasonal_forecast'],
    Field(
        description='Type of simulation: historical (hindcast), seasonal retrospective forecast '
        '(reforecast), or real-time seasonal forecast'
    ),
]
Frequency = Annotated[
    Literal['daily', 'monthly'],
    Field(description='Averaging frequency of the output (daily or monthly averages)'),
]


def validate_catalog_file_path(file_path: str | Path) -> Path:
    """
    Validate that the catalog file path is within the project directory.

    Args:
        file_path: Path to validate

    Returns:
        Path: Validated and resolved path

    Raises:
        ValueError: If path is outside project directory or contains path traversal
    """
    # Get the project root directory (where this file is located)
    # In newer python the resolve method defaults to strict=False
    # which will not raise file not found errors.
    project_root = Path(__file__).parents[2].resolve()

    # Convert to Path and resolve to absolute path
    resolved_path = Path(file_path).resolve()

    # Check if the resolved path is within the project directory
    try:
        resolved_path.relative_to(project_root)
    except ValueError:
        raise ValueError(
            f'Access denied: File path {resolved_path} is forbidden {project_root}'
        ) from None

    # Additional check for common path traversal patterns
    str_path = str(file_path)
    if '..' in str_path or str_path.startswith('/'):
        raise ValueError(f'Access denied: Path contains forbidden patterns: {file_path}')

    return resolved_path


class CEFIDatasetMetadata(BaseModel):
    """CEFI dataset metadata, including variable name and units, frequency,
    coordinate ranges, and geographic region.
    """

    # Fields that are commented out exist in every CEFI dataset,
    # but are currently not considered relevant for the MCP server.
    # cefi_filename: str = Field(..., description='CEFI standardized filename')
    cefi_variable: str = Field(..., description='Variable name in the dataset')
    cefi_long_name: str = Field(..., description='Human-readable variable description')
    cefi_unit: str = Field(..., description='Variable units')
    cefi_output_frequency: Literal['daily', 'monthly'] = Field(
        ..., description='Temporal frequency of output (daily or monthly)'
    )
    cefi_grid_type: str = Field(
        ...,
        description='Grid type (raw or regridded). The regridded data will be easier to work with.',
    )
    # cefi_rel_path: str = Field(..., description='Relative path within CEFI archive')
    # cefi_ori_filename: str = Field(..., description='Original filename before CEFI processing')
    # cefi_archive_version: str = Field(..., description='Archive version path')
    # cefi_run_xml: str = Field(..., description='Run XML configuration')
    cefi_region: Literal['nwa', 'nep'] = Field(..., description='Geographic region code')
    cefi_subdomain: str = Field(..., description='Subdomain within region')
    cefi_experiment_type: Literal['hindcast', 'seasonal_reforecast', 'seasonal_forecast'] = Field(
        ..., description='Type of experiment (hindcast, forecast, etc.)'
    )
    # cefi_experiment_name: str = Field(..., description='Name of the experiment')
    cefi_release: str = Field(..., description='CEFI release version')
    cefi_date_range: str = Field(..., description='Date range covered by data')
    cefi_init_date: str = Field(..., description='Initialization date')
    cefi_ensemble_info: str = Field(..., description='Ensemble member information')
    cefi_forcing: str = Field(..., description='Forcing data information')
    cefi_data_doi: str = Field(..., description='DOI for the dataset')
    cefi_paper_doi: str = Field(..., description='DOI for associated paper')
    cefi_aux: str = Field(..., description='Auxiliary information')
    # cefi_ori_category: str = Field(..., description='Original data category')
    cefi_opendap: HttpUrl = Field(..., description='OpenDAP URL for data access')

    @computed_field
    @property
    def start_date(self) -> str | NaiveDatetime:
        """The first time available in the dataset"""
        if self.cefi_date_range == 'N/A':
            return 'N/A'
        else:
            return datetime.strptime(self.cefi_date_range[0:6] + '01', '%Y%m%d')

    @computed_field
    @property
    def end_date(self) -> str | NaiveDatetime:
        """The last time available in the dataset"""
        if self.cefi_date_range == 'N/A':
            return 'N/A'
        else:
            return datetime.strptime(self.cefi_date_range[7:13] + '31', '%Y%m%d')

    @computed_field
    @property
    def release_date(self) -> NaiveDatetime:
        """The date when the dataset was published. Newer dates are preferred."""
        return datetime.strptime(self.cefi_release[1:9], '%Y%m%d')

    @computed_field
    @property
    def init_date(self) -> str | NaiveDatetime:
        """The date when the dataset was published. Newer dates are generally preferred."""
        if self.cefi_init_date == 'N/A':
            return 'N/A'
        else:
            return datetime.strptime(self.cefi_init_date[1:7] + '01', '%Y%m%d')


class CEFIDataCatalog(BaseModel):
    """Model for CEFI data catalog containing multiple datasets"""

    datasets: dict[str, CEFIDatasetMetadata] = Field(
        ..., description='Dictionary of dataset metadata keyed by dataset ID'
    )

    def get_dataset_from_catalog(self, dataset_id: str) -> CEFIDatasetMetadata | None:
        """Get the full catalog listing for a specific dataset by the dataset ID"""
        return self.datasets.get(dataset_id)

    def list_variables(self) -> dict[str, str]:
        """
        List all unique variables in the CEFI data catalog,
        including their abbreviation and human-readable name.
        """
        return {dataset.cefi_variable: dataset.cefi_long_name for dataset in self.datasets.values()}

    def filter_by_variable(self, variable: str) -> dict[str, CEFIDatasetMetadata]:
        """Filter datasets by variable name"""
        return {
            dataset_id: dataset
            for dataset_id, dataset in self.datasets.items()
            if dataset.cefi_variable == variable
        }

    def filter_by_frequency(self, frequency: Frequency) -> dict[str, CEFIDatasetMetadata]:
        """Filter datasets by output frequency"""
        return {
            dataset_id: dataset
            for dataset_id, dataset in self.datasets.items()
            if dataset.cefi_output_frequency == frequency
        }


async def load_catalog_file(json_file: str | Path) -> CEFIDataCatalog:
    """
    Load the catalog from a local .json file.
    Primarily for local testing during development.

    Args:
        json_file: Path to the JSON catalog file (must be within project directory)

    Returns:
        CEFIDataCatalog: The loaded catalog

    Raises:
        ValueError: If the file path is outside the project directory
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    # Validate that the file path is within the project directory
    validated_path = validate_catalog_file_path(json_file)
    logger.info('Loading catalog from file: {file}', file=validated_path)
    # Use async file operations
    async with aiofiles.open(validated_path) as f:
        content = await f.read()
        data = await asyncio.to_thread(json.loads, content)

    # Convert the JSON structure to our model format
    datasets = {}
    for dataset_id, dataset_info in data.items():
        datasets[dataset_id] = CEFIDatasetMetadata(**dataset_info)
    logger.info('Loaded {n} datasets from local file', n=len(datasets))
    return CEFIDataCatalog(datasets=datasets)


async def fetch_json(url: str, session: aiohttp.ClientSession | None = None) -> dict:
    """
    Fetch a JSON document from `url` asynchronously and return it as a dict.
    """
    close_session = False
    if session is None:
        logger.debug('Starting aiohttp.ClientSession')
        session = aiohttp.ClientSession()
        close_session = True
    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as e:
        logger.error('Failed to fetch JSON from {url}: {error}', url=url, error=str(e))
        raise
    finally:
        if close_session:
            logger.debug('Closing session')
            await session.close()


@alru_cache(maxsize=3)
async def load_catalog_remote(
    region: Region,
    simulation: Simulation,
) -> CEFIDataCatalog:
    url = f'https://psl.noaa.gov/cefi_portal/data_index/cefi_data_indexing.Projects.CEFI.regional_mom6.cefi_portal.{region}.full_domain.{simulation}.json'
    logger.info('Fetching catalog from {url}', url=url)
    data = await fetch_json(url)
    # Convert the JSON structure to our model format
    datasets = {}
    for dataset_id, dataset_info in data.items():
        datasets[dataset_id] = CEFIDatasetMetadata(**dataset_info)
    logger.info('Loaded {n} datasets from remote file', n=len(datasets))
    return CEFIDataCatalog(datasets=datasets)


async def get_dataset_from_catalog(
    region: Region,
    simulation: Simulation,
    dataset_id: str,
) -> CEFIDatasetMetadata | None:
    """Get the full catalog listing for a specific dataset by the dataset ID"""
    logger.info(
        'Getting dataset {dataset_id} for {region}/{simulation}',
        dataset_id=dataset_id,
        region=region,
        simulation=simulation,
    )
    cat = await load_catalog_remote(region, simulation)
    return cat.get_dataset_from_catalog(dataset_id)


async def list_variables(
    region: Region,
    simulation: Simulation,
) -> dict[str, str]:
    """List all unique variables in the CEFI data catalog"""
    logger.info('Listing variables for {region}/{simulation}', region=region, simulation=simulation)
    cat = await load_catalog_remote(region, simulation)
    return cat.list_variables()


async def filter_by_variable(
    region: Region,
    simulation: Simulation,
    variable: Variable,
) -> dict[str, CEFIDatasetMetadata]:
    """Filter datasets by variable name"""
    logger.info(
        'Filtering datasets by variable {variable} for {region}/{simulation}',
        variable=variable,
        region=region,
        simulation=simulation,
    )
    cat = await load_catalog_remote(region, simulation)
    return cat.filter_by_variable(variable)


async def filter_by_frequency(
    region: Region,
    simulation: Simulation,
    frequency: Frequency,
) -> dict[str, CEFIDatasetMetadata]:
    """Filter datasets by output frequency"""
    logger.info(
        'Filtering datasets by frequency {frequency} for {region}/{simulation}',
        frequency=frequency,
        region=region,
        simulation=simulation,
    )
    cat = await load_catalog_remote(region, simulation)
    return cat.filter_by_frequency(frequency)
