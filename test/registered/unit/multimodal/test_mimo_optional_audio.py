"""MiMo processor discovery must not require the optional audio decoder."""

import builtins
import importlib
import sys
import unittest
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_AUDIO = "sglang.srt.multimodal.processors.mimo_audio"
_PROCESSOR = "sglang.srt.multimodal.processors.mimo_v2"


class TestMiMoOptionalAudio(unittest.TestCase):
    def _load_without_torchcodec(self, error):
        original_import = builtins.__import__

        def without_torchcodec(name, *args, **kwargs):
            if name == "torchcodec" or name.startswith("torchcodec."):
                raise error
            return original_import(name, *args, **kwargs)

        # Restore both the import cache and parent-package attributes so this
        # dependency simulation cannot affect later tests in the same process.
        package = importlib.import_module("sglang.srt.multimodal.processors")
        with patch.dict(sys.modules), patch.dict(package.__dict__):
            sys.modules.pop(_AUDIO, None)
            sys.modules.pop(_PROCESSOR, None)
            with patch("builtins.__import__", side_effect=without_torchcodec):
                audio = importlib.import_module(_AUDIO)
                processor = importlib.import_module(_PROCESSOR)
        return audio, processor

    def test_processor_imports_without_working_torchcodec(self):
        for error in (
            ModuleNotFoundError("No module named 'torchcodec'"),
            RuntimeError("Could not load libtorchcodec"),
            OSError("Could not load FFmpeg shared libraries"),
        ):
            with self.subTest(error=type(error).__name__):
                audio, processor = self._load_without_torchcodec(error)
                self.assertIsNone(audio.AudioDecoder)
                self.assertIsNone(processor.AudioDecoder)
                self.assertIn(
                    "MiMoV2ForCausalLM",
                    [model.__name__ for model in processor.MiMoV2Processor.models],
                )
                # Constructing the shared pipeline does not decode audio.
                audio.MiMoAudioPipeline(
                    audio_token_id=1, audio_start_token_id=2, audio_end_token_id=3
                )

    def test_missing_decoder_does_not_silently_drop_video_audio(self):
        _, processor = self._load_without_torchcodec(ModuleNotFoundError())
        with (
            patch.object(processor, "download_remote_media") as download,
            self.assertRaisesRegex(RuntimeError, "torchcodec.*audio tracks"),
        ):
            processor.MiMoProcessor.has_audio_track("https://example.com/video.mp4")
        download.assert_not_called()

    def test_encoded_audio_reports_missing_decoder(self):
        audio, _ = self._load_without_torchcodec(ModuleNotFoundError())
        pipeline = audio.MiMoAudioPipeline(
            audio_token_id=1, audio_start_token_id=2, audio_end_token_id=3
        )
        # Isolate TorchCodec from the separate optional torchaudio dependency.
        with (
            patch.object(pipeline, "_ensure_audio_dependencies"),
            self.assertRaisesRegex(RuntimeError, "torchcodec.*audio decoding"),
        ):
            pipeline.preprocess_audio(b"encoded audio")

    def test_working_decoder_probes_downloaded_bytes(self):
        processor = importlib.import_module(_PROCESSOR)
        decoder = Mock()
        with (
            patch.object(processor, "AudioDecoder", decoder),
            patch.object(
                processor, "download_remote_media", return_value=b"video with audio"
            ) as download,
        ):
            self.assertTrue(
                processor.MiMoProcessor.has_audio_track("https://example.com/video.mp4")
            )
        download.assert_called_once()
        self.assertEqual(decoder.call_args.args[0].read(), b"video with audio")

    def test_video_without_audio_still_returns_false(self):
        processor = importlib.import_module(_PROCESSOR)
        with patch.object(
            processor, "AudioDecoder", side_effect=RuntimeError("No audio stream")
        ):
            self.assertFalse(processor.MiMoProcessor.has_audio_track(b"silent video"))


if __name__ == "__main__":
    unittest.main()
