"""Local assets must not enter the downloader, including Windows drive paths."""

import pytest

from openpi.shared import download


@pytest.mark.parametrize("style", ["native", "posix", "path_object"])
def test_local_assets_bypass_remote_cache(tmp_path, monkeypatch, style):
    def unexpected_cache():
        pytest.fail("Local assets must not use the remote cache")

    monkeypatch.setattr(download, "get_cache_dir", unexpected_cache)
    asset = tmp_path / "norm_stats.json"
    url = {"native": str(asset), "posix": asset.as_posix(), "path_object": asset}[style]
    with pytest.raises(FileNotFoundError, match="File not found"):
        download.maybe_download(url)
    asset.write_text("{}", encoding="utf-8")
    assert download.maybe_download(url) == asset.resolve()
