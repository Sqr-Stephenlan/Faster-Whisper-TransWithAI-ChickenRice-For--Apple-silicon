import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

import download_models


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error")

    def iter_content(self, chunk_size: int):
        yield self.content

    def close(self) -> None:
        self.closed = True


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.calls: list[tuple[str, bool, int]] = []

    def get(self, url: str, stream: bool, timeout: int) -> FakeResponse:
        self.calls.append((url, stream, timeout))
        return self.responses.pop(0)


class DownloadFileRetryTests(unittest.TestCase):
    def test_retries_429_then_downloads_file(self) -> None:
        first_response = FakeResponse(429)
        second_response = FakeResponse(200, b"model-data", {"content-length": "10"})
        session = FakeSession([first_response, second_response])

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = Path(tmp_dir) / "model.onnx"
            with patch("download_models.time.sleep") as sleep:
                self.assertTrue(download_models.download_file("https://example.test/model.onnx", dest_path, session))

            self.assertEqual(dest_path.read_bytes(), b"model-data")

        self.assertTrue(first_response.closed)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(
            session.calls,
            [
                ("https://example.test/model.onnx", True, 30),
                ("https://example.test/model.onnx", True, 30),
            ],
        )

    def test_uses_retry_after_header_for_429_delay(self) -> None:
        session = FakeSession(
            [
                FakeResponse(429, headers={"Retry-After": "3"}),
                FakeResponse(200, b"ok", {"content-length": "2"}),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch("download_models.time.sleep") as sleep:
            self.assertTrue(
                download_models.download_file(
                    "https://example.test/config.json", Path(tmp_dir) / "config.json", session
                )
            )

        sleep.assert_called_once_with(3.0)

    def test_non_429_error_is_not_retried(self) -> None:
        session = FakeSession([FakeResponse(404)])

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertFalse(
                download_models.download_file(
                    "https://example.test/missing.json", Path(tmp_dir) / "missing.json", session
                )
            )

        self.assertEqual(len(session.calls), 1)

    def test_exhausted_429_retries_fail_and_remove_partial_file(self) -> None:
        responses = [FakeResponse(429) for _ in range(download_models.DOWNLOAD_MAX_RETRIES + 1)]
        session = FakeSession(responses)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = Path(tmp_dir) / "model.onnx"
            with patch("download_models.time.sleep"):
                self.assertFalse(download_models.download_file("https://example.test/model.onnx", dest_path, session))

            self.assertFalse(dest_path.exists())

        self.assertEqual(len(session.calls), download_models.DOWNLOAD_MAX_RETRIES + 1)


class DownloadFileMirrorTests(unittest.TestCase):
    PRIMARY_URL = "https://huggingface.co/owner/repo/resolve/main/model.onnx"
    MIRROR_URL = "https://hf-mirror.com/owner/repo/resolve/main/model.onnx"

    def test_falls_back_to_hf_mirror_after_primary_exhausts_retries(self) -> None:
        primary_responses = [FakeResponse(429) for _ in range(download_models.DOWNLOAD_MAX_RETRIES + 1)]
        mirror_response = FakeResponse(200, b"mirror-data", {"content-length": "11"})
        session = FakeSession([*primary_responses, mirror_response])

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = Path(tmp_dir) / "model.onnx"
            with patch("download_models.time.sleep"):
                self.assertTrue(download_models.download_file(self.PRIMARY_URL, dest_path, session))

            self.assertEqual(dest_path.read_bytes(), b"mirror-data")

        called_urls = [call[0] for call in session.calls]
        self.assertEqual(called_urls.count(self.PRIMARY_URL), download_models.DOWNLOAD_MAX_RETRIES + 1)
        self.assertEqual(called_urls[-1], self.MIRROR_URL)

    def test_falls_back_to_hf_mirror_on_non_429_error(self) -> None:
        session = FakeSession(
            [
                FakeResponse(500),
                FakeResponse(200, b"ok", {"content-length": "2"}),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = Path(tmp_dir) / "model.onnx"
            self.assertTrue(download_models.download_file(self.PRIMARY_URL, dest_path, session))
            self.assertEqual(dest_path.read_bytes(), b"ok")

        self.assertEqual([call[0] for call in session.calls], [self.PRIMARY_URL, self.MIRROR_URL])

    def test_mirror_failure_removes_partial_file_after_both_hosts_exhausted(self) -> None:
        responses = [FakeResponse(429) for _ in range(2 * (download_models.DOWNLOAD_MAX_RETRIES + 1))]
        session = FakeSession(responses)

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = Path(tmp_dir) / "model.onnx"
            with patch("download_models.time.sleep"):
                self.assertFalse(download_models.download_file(self.PRIMARY_URL, dest_path, session))

            self.assertFalse(dest_path.exists())

        self.assertEqual(len(session.calls), 2 * (download_models.DOWNLOAD_MAX_RETRIES + 1))

    def test_non_huggingface_url_has_no_mirror_fallback(self) -> None:
        session = FakeSession([FakeResponse(500)])

        with tempfile.TemporaryDirectory() as tmp_dir:
            dest_path = Path(tmp_dir) / "model.onnx"
            self.assertFalse(download_models.download_file("https://example.test/model.onnx", dest_path, session))

        self.assertEqual(len(session.calls), 1)


class StrictDownloadTests(unittest.TestCase):
    def test_hf_model_partial_download_returns_false(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(download_models, "MODELS_ROOT", Path(tmp_dir)),
            patch.object(download_models, "get_hf_api_files", return_value=["config.json", "model.bin"]),
            patch.object(download_models, "download_file", side_effect=[True, False]),
        ):
            self.assertFalse(download_models.download_hf_model("owner/model", "translate"))

    def test_verify_only_never_calls_network(self) -> None:
        with (
            patch("sys.argv", ["download_models.py", "--profile", "all", "--verify-only", "--non-interactive"]),
            patch.object(download_models.requests, "get") as get,
            patch.object(download_models.requests, "Session") as session,
        ):
            self.assertEqual(download_models.main(), 1)

        get.assert_not_called()
        session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
