import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from faster_whisper_transwithai_chickenrice.backends.base import (
    BackendRequest,
    BackendUnavailableError,
    ModelDescriptor,
    UnsupportedBackendOptionError,
)
from faster_whisper_transwithai_chickenrice.backends.factory import (
    BackendAvailability,
    probe_backend,
    select_backend,
)
from faster_whisper_transwithai_chickenrice.backends.mlx import MLXBackend
from faster_whisper_transwithai_chickenrice.backends.option_mapping import (
    map_ct2_options,
    map_mlx_options,
)
from faster_whisper_transwithai_chickenrice.profiles import get_profile


class OptionMappingTests(unittest.TestCase):
    def test_ct2_keeps_supported_options_and_removes_request_fields(self) -> None:
        mapped = map_ct2_options(
            {
                "language": "ja",
                "task": "translate",
                "beam_size": 1,
                "repetition_penalty": 1.1,
                "smart_split_with_vad": True,
            }
        )

        self.assertEqual(mapped, {"beam_size": 1, "repetition_penalty": 1.1})

    def test_mlx_beam_size_one_uses_greedy_compatibility_mapping(self) -> None:
        mapping = map_mlx_options(
            {
                "beam_size": 1,
                "condition_on_previous_text": False,
                "repetition_penalty": 1.1,
                "vad_filter": True,
            },
            task="translate",
        )

        self.assertNotIn("beam_size", mapping.options)
        self.assertTrue(mapping.options["fp16"])
        self.assertFalse(mapping.options["word_timestamps"])
        self.assertEqual(mapping.ignored, ("repetition_penalty", "vad_filter"))

    def test_mlx_rejects_unimplemented_beam_search(self) -> None:
        with self.assertRaisesRegex(UnsupportedBackendOptionError, "beam search"):
            map_mlx_options({"beam_size": 2}, task="translate")

    def test_mlx_strict_mode_rejects_unknown_options(self) -> None:
        with self.assertRaisesRegex(UnsupportedBackendOptionError, "unknown_option"):
            map_mlx_options({"unknown_option": True}, task="translate", strict_unknown=True)


class MLXBackendAdapterTests(unittest.TestCase):
    def test_normalizes_dictionary_result_without_runtime_import(self) -> None:
        transcribe = mock.Mock(
            return_value={
                "language": "ja",
                "segments": [
                    {"start": 0.25, "end": 1.5, "text": " 中文 "},
                    {"start": 1.5, "end": 1.5, "text": "empty"},
                ],
            }
        )
        backend = MLXBackend(
            ModelDescriptor(
                backend="mlx",
                profile="translate",
                variant="fp16",
                path=Path("/models/mlx/translate/fp16"),
            ),
            transcribe_fn=transcribe,
        )

        result = backend.transcribe(
            BackendRequest(
                audio=[0.0] * 32_000,
                language="ja",
                task="translate",
                options={"beam_size": 1, "repetition_penalty": 1.1},
            )
        )

        self.assertEqual(result.backend, "mlx")
        self.assertEqual(result.duration, 2.0)
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].text, "中文")
        kwargs = transcribe.call_args.kwargs
        self.assertNotIn("beam_size", kwargs)
        self.assertNotIn("repetition_penalty", kwargs)
        self.assertEqual(kwargs["path_or_hf_repo"], "/models/mlx/translate/fp16")


class BackendFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = get_profile("translate")
        self.mlx_descriptor = self.profile.descriptor("mlx")
        self.ct2_descriptor = self.profile.descriptor("ct2")

    def test_auto_falls_back_only_during_preflight(self) -> None:
        unavailable_mlx = BackendAvailability(
            backend="mlx",
            available=False,
            descriptor=self.mlx_descriptor,
            reasons=("Metal unavailable",),
        )
        available_ct2 = BackendAvailability(
            backend="ct2",
            available=True,
            descriptor=self.ct2_descriptor,
            device="cpu",
        )
        with mock.patch(
            "faster_whisper_transwithai_chickenrice.backends.factory.probe_backend",
            side_effect=[unavailable_mlx, available_ct2],
        ):
            selection = select_backend("auto", self.profile)

        self.assertEqual(selection.selected, "ct2")
        self.assertIn("Metal unavailable", selection.fallback_reason)

    def test_explicit_mlx_never_falls_back_to_ct2(self) -> None:
        unavailable_mlx = BackendAvailability(
            backend="mlx",
            available=False,
            descriptor=self.mlx_descriptor,
            reasons=("Metal unavailable",),
        )
        with (
            mock.patch(
                "faster_whisper_transwithai_chickenrice.backends.factory.probe_backend",
                return_value=unavailable_mlx,
            ) as probe,
            self.assertRaisesRegex(BackendUnavailableError, "Metal unavailable"),
        ):
            select_backend("mlx", self.profile)

        self.assertEqual(probe.call_count, 1)

    def test_probe_without_runtime_check_is_ci_safe(self) -> None:
        with (
            mock.patch(
                "faster_whisper_transwithai_chickenrice.backends.factory.validate_mlx_model",
                return_value=([], []),
            ),
            mock.patch(
                "faster_whisper_transwithai_chickenrice.backends.factory.platform.machine",
                return_value="arm64",
            ),
        ):
            availability = probe_backend(
                self.profile,
                "mlx",
                check_runtime=False,
                verify_hashes=False,
            )

        self.assertTrue(availability.available)
        self.assertEqual(availability.device, "gpu")


class ProfileSchemaTests(unittest.TestCase):
    def test_transcribe_assets_are_isolated_from_translate_assets(self) -> None:
        translate = get_profile("translate").descriptor("mlx")
        transcribe = get_profile("transcribe").descriptor("mlx")

        self.assertNotEqual(translate.path, transcribe.path)
        self.assertIn("translate", translate.path.parts)
        self.assertIn("transcribe", transcribe.path.parts)


if __name__ == "__main__":
    unittest.main()
