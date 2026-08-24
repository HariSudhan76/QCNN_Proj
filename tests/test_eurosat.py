import ssl
from unittest.mock import patch
from urllib.error import URLError

import pytest

from qrs.data.eurosat import download_eurosat


def _make_ssl_cert_url_error() -> URLError:
    """Reproduce exactly what urlretrieve raises in practice: urlopen catches
    the ssl.SSLCertVerificationError and re-raises it wrapped in a URLError,
    with the original exception as `.reason` -- not the SSLCertVerificationError
    itself. A prior version of the fallback caught the wrong type and never
    triggered because of this."""
    cert_err = ssl.SSLCertVerificationError(
        "certificate verify failed: unable to get local issuer certificate"
    )
    return URLError(cert_err)


def test_falls_back_to_unverified_on_ssl_cert_error(tmp_path):
    calls = {"n": 0}

    def fake_urlretrieve(url, path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_ssl_cert_url_error()
        path.write_bytes(b"fake zip bytes")

    with patch("qrs.data.eurosat.urlretrieve", side_effect=fake_urlretrieve):
        with patch("qrs.data.eurosat.zipfile.ZipFile"):
            with pytest.warns(UserWarning, match="WITHOUT certificate verification"):
                download_eurosat(tmp_path)

    assert calls["n"] == 2  # first attempt failed, fallback attempt succeeded


def test_non_ssl_url_error_is_not_swallowed(tmp_path):
    def fake_urlretrieve(url, path):
        raise URLError("name resolution failed")

    with patch("qrs.data.eurosat.urlretrieve", side_effect=fake_urlretrieve):
        with pytest.raises(URLError):
            download_eurosat(tmp_path)
