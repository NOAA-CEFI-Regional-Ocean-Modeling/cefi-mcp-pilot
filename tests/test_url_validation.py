import pytest

from src.cefi_mcp.tools import validate_cefi_opendap_url


def test_valid_cefi_urls():
    """Test that valid CEFI OpenDAP URLs are allowed."""
    valid_urls = [
        'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/regional_mom6/cefi_portal/northwest_atlantic/full_domain/hindcast/daily/raw/r20230520/btm_co3_ion.nwa.full.hcast.daily.raw.r20230520.199301-201912.nc',
        'https://psl.noaa.gov/thredds/dodsC/Projects/CEFI/test.nc',
        'http://psl.noaa.gov/thredds/dodsC/Projects/CEFI/seasonal_forecast/monthly/raw/data.nc',
    ]

    for url in valid_urls:
        assert validate_cefi_opendap_url(url) is True, f'Valid URL should be allowed: {url}'


def test_invalid_domain():
    """Test that URLs with invalid domains are rejected."""
    invalid_urls = [
        'http://evil.com/thredds/dodsC/Projects/CEFI/data.nc',
        'http://subdomain.psl.noaa.gov/thredds/dodsC/Projects/CEFI/data.nc',
        'http://psl.noaa.com/thredds/dodsC/Projects/CEFI/data.nc',
        'http://localhost:8080/admin',
        'http://169.254.169.254/latest/meta-data/',
    ]

    for url in invalid_urls:
        with pytest.raises(ValueError, match='Invalid domain'):
            validate_cefi_opendap_url(url)


def test_invalid_protocol():
    """Test that URLs with invalid protocols are rejected."""
    invalid_urls = [
        'ftp://psl.noaa.gov/thredds/dodsC/Projects/CEFI/data.nc',
        'file://psl.noaa.gov/thredds/dodsC/Projects/CEFI/data.nc',
        'javascript://psl.noaa.gov/thredds/dodsC/Projects/CEFI/data.nc',
    ]

    for url in invalid_urls:
        with pytest.raises(ValueError, match='Invalid URL scheme'):
            validate_cefi_opendap_url(url)


def test_invalid_path():
    """Test that URLs with invalid paths are rejected."""
    invalid_urls = [
        'http://psl.noaa.gov/other/path/data.nc',
        'http://psl.noaa.gov/thredds/dodsC/Projects/OTHER/data.nc',
        'http://psl.noaa.gov/thredds/dodsC/Projects/',
        'http://psl.noaa.gov/admin',
        'http://psl.noaa.gov/',
    ]

    for url in invalid_urls:
        with pytest.raises(ValueError, match='Invalid path'):
            validate_cefi_opendap_url(url)


def test_ssrf_prevention():
    """Test that common SSRF attack URLs are blocked."""
    ssrf_urls = [
        'http://localhost:8080/admin',
        'http://127.0.0.1:8080/',
        'http://169.254.169.254/latest/meta-data/',
        'http://metadata.google.internal/',
        'http://[::1]:8080/',
    ]

    for url in ssrf_urls:
        with pytest.raises(ValueError):
            validate_cefi_opendap_url(url)


def test_malformed_urls():
    """Test that malformed URLs are rejected."""
    malformed_urls = [
        'not-a-url',
        'http://',
        '://psl.noaa.gov/path',
        '',
        'http:///path',
    ]

    for url in malformed_urls:
        with pytest.raises(ValueError):
            validate_cefi_opendap_url(url)
