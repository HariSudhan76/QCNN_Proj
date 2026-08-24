import io
import ssl
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest
import truststore

from qrs.data.eurosat import download_eurosat


def test_truststore_injection_actually_replaces_sslcontext():
    """Ground-truth check (no mocking) that truststore.inject_into_ssl()
    really does swap ssl.SSLContext for one that forces OS-backed
    verification, and extract_from_ssl() really does restore the stdlib
    class -- the premise the fallback in download_eurosat depends on."""
    # Importing qrs.data.eurosat already called inject_into_ssl() at module
    # load time, so extract first to get a genuine reference to the stdlib
    # class before testing the round-trip.
    truststore.extract_from_ssl()
    stdlib_context_class = ssl.SSLContext

    truststore.inject_into_ssl()
    try:
        assert ssl.SSLContext is not stdlib_context_class
    finally:
        truststore.extract_from_ssl()

    assert ssl.SSLContext is stdlib_context_class

    # Restore the global state eurosat.py's module-level call establishes,
    # so later tests aren't affected by this test having run.
    truststore.inject_into_ssl()


def test_truststore_context_ignores_verify_mode_when_injected():
    """Ground-truth check that an "unverified" context built while truststore
    is still injected is NOT actually bypassable: truststore.SSLContext's
    wrap_socket calls _verify_peercerts() unconditionally, regardless of
    verify_mode. This is why the fallback must extract_from_ssl() first
    rather than just building ssl._create_unverified_context()."""
    truststore.inject_into_ssl()
    try:
        ctx = ssl._create_unverified_context()
        assert isinstance(ctx, truststore.SSLContext)
        assert ctx.verify_mode == ssl.CERT_NONE  # looks unverified...
        # ...but wrap_socket ignores that. We don't open a real (bad-cert)
        # socket here -- that's what the extract_from_ssl round-trip test
        # above establishes indirectly -- this just documents the trap.
    finally:
        truststore.extract_from_ssl()
        truststore.inject_into_ssl()


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


def _raise_ssl_cert_url_error(*args, **kwargs):
    raise _make_ssl_cert_url_error()


def _make_fake_opener(content: bytes) -> MagicMock:
    """A stand-in for build_opener(...)'s return value, whose .open() acts as
    a context manager yielding a file-like response."""
    response = io.BytesIO(content)
    response.__enter__ = lambda self=response: self
    response.__exit__ = lambda self, *exc: False
    opener = MagicMock()
    opener.open.return_value = response
    return opener


def test_falls_back_to_unverified_on_ssl_cert_error(tmp_path):
    fake_opener = _make_fake_opener(b"fake zip bytes")

    with patch("qrs.data.eurosat.urlretrieve", side_effect=_raise_ssl_cert_url_error):
        with patch("qrs.data.eurosat.build_opener", return_value=fake_opener) as build_opener_mock:
            with patch("qrs.data.eurosat.zipfile.ZipFile"):
                with pytest.warns(UserWarning, match="WITHOUT certificate verification"):
                    download_eurosat(tmp_path)

    build_opener_mock.assert_called_once()
    fake_opener.open.assert_called_once()


def test_fallback_temporarily_undoes_truststore_injection(tmp_path):
    """truststore.inject_into_ssl() replaces ssl.SSLContext globally with one
    whose wrap_socket always forces OS-backed verification regardless of
    verify_mode -- so the fallback must extract_from_ssl() before building
    its "unverified" context and inject_into_ssl() again afterward, or the
    retry fails identically to the first attempt."""
    trace: list[str] = []
    fake_opener = _make_fake_opener(b"fake zip bytes")

    def fake_build_opener(*args, **kwargs):
        # By the time the fallback builds its opener, truststore's
        # injection must already be undone.
        assert trace == ["extract"], trace
        return fake_opener

    with patch("qrs.data.eurosat.urlretrieve", side_effect=_raise_ssl_cert_url_error):
        with patch("qrs.data.eurosat.build_opener", side_effect=fake_build_opener):
            with patch("qrs.data.eurosat.zipfile.ZipFile"):
                with patch(
                    "qrs.data.eurosat.truststore.extract_from_ssl",
                    side_effect=lambda: trace.append("extract"),
                ) as extract_mock:
                    with patch(
                        "qrs.data.eurosat.truststore.inject_into_ssl",
                        side_effect=lambda: trace.append("inject"),
                    ) as inject_mock:
                        with pytest.warns(UserWarning):
                            download_eurosat(tmp_path)

    extract_mock.assert_called_once()
    inject_mock.assert_called_once()
    assert trace == ["extract", "inject"]  # injection restored afterward


def test_non_ssl_url_error_is_not_swallowed(tmp_path):
    def fake_urlretrieve(url, path):
        raise URLError("name resolution failed")

    with patch("qrs.data.eurosat.urlretrieve", side_effect=fake_urlretrieve):
        with pytest.raises(URLError):
            download_eurosat(tmp_path)
