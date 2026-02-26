"""Silero VAD wrapper for speech detection.

Uses Silero VAD v5 ONNX model (~2MB) for accurate speech/silence detection.
Falls back gracefully if onnxruntime is not available.

Usage:
    vad = SileroVAD(threshold=0.5, min_speech_ms=300, min_silence_ms=700)
    if vad.available:
        result = vad.process_chunk(audio_f32_16khz)
        # result.is_speech, result.speech_ended, result.speech_audio

Author: Metis CEO (2026-02-24)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Silero VAD expects 16kHz mono, chunks of 512 samples (32ms)
SILERO_SAMPLE_RATE = 16000
SILERO_CHUNK_SAMPLES = 512


@dataclass
class VADResult:
    """Result of processing one audio chunk."""
    is_speech: bool = False
    speech_started: bool = False  # First speech chunk detected (for DoA)
    speech_ended: bool = False
    speech_audio: Optional[bytes] = None  # Accumulated PCM if speech_ended


class SileroVAD:
    """Silero VAD v5 wrapper with speech accumulation.

    Args:
        threshold: Speech probability threshold (0.0-1.0). Default 0.5.
        min_speech_ms: Minimum speech duration to accept (ms). Default 300.
        min_silence_ms: Silence duration to mark speech end (ms). Default 700.
        sample_rate: Audio sample rate. Default 16000.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 300,
        min_silence_ms: int = 700,
        sample_rate: int = SILERO_SAMPLE_RATE,
    ) -> None:
        self.threshold = threshold
        self.min_speech_samples = int(min_speech_ms * sample_rate / 1000)
        self.min_silence_samples = int(min_silence_ms * sample_rate / 1000)
        self.sample_rate = sample_rate

        self._model: object | None = None
        self._available = False
        self._load_attempted = False

        # Internal state
        self._is_speech = False
        self._speech_buffer = bytearray()
        self._speech_samples = 0
        self._silence_samples = 0
        self._chunk_buffer = np.array([], dtype=np.float32)

    @property
    def available(self) -> bool:
        """Check if Silero VAD can be used. Triggers lazy load on first access."""
        if not self._load_attempted:
            self._load_model()
        return self._available

    def _load_model(self) -> None:
        """Load Silero VAD ONNX model via torch.hub (called lazily on first use)."""
        self._load_attempted = True
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            self._model = model
            self._available = True
            logger.info("Silero VAD loaded (ONNX)")
        except Exception as e:
            logger.warning("Silero VAD not available, will use energy-based fallback: %s", e)
            self._available = False

    def reset(self) -> None:
        """Reset internal state for new session."""
        self._is_speech = False
        self._speech_buffer = bytearray()
        self._speech_samples = 0
        self._silence_samples = 0
        self._chunk_buffer = np.array([], dtype=np.float32)
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()

    def process_chunk(self, audio_f32: NDArray[np.float32]) -> VADResult:
        """Process an audio chunk and return VAD result.

        Args:
            audio_f32: Float32 audio samples [-1.0, 1.0], 16kHz mono.

        Returns:
            VADResult with speech detection status.
        """
        if not self._load_attempted:
            self._load_model()
        if not self._available:
            return VADResult()

        import torch

        # Accumulate into chunk buffer for 512-sample processing
        self._chunk_buffer = np.concatenate([self._chunk_buffer, audio_f32])

        result = VADResult()

        # Process in 512-sample chunks (Silero requirement)
        while len(self._chunk_buffer) >= SILERO_CHUNK_SAMPLES:
            chunk = self._chunk_buffer[:SILERO_CHUNK_SAMPLES]
            self._chunk_buffer = self._chunk_buffer[SILERO_CHUNK_SAMPLES:]

            # Run Silero inference
            tensor = torch.from_numpy(chunk)
            prob = float(self._model(tensor, self.sample_rate))

            chunk_i16 = np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)

            if prob >= self.threshold:
                # Speech detected
                if not self._is_speech:
                    self._is_speech = True
                    self._speech_buffer = bytearray()
                    self._speech_samples = 0
                    result.speech_started = True  # First frame of new speech
                self._silence_samples = 0
                self._speech_buffer.extend(chunk_i16.tobytes())
                self._speech_samples += SILERO_CHUNK_SAMPLES
                result.is_speech = True

            elif self._is_speech:
                # Was speaking, now silence
                self._speech_buffer.extend(chunk_i16.tobytes())
                self._silence_samples += SILERO_CHUNK_SAMPLES

                if self._silence_samples >= self.min_silence_samples:
                    # Speech ended
                    if self._speech_samples >= self.min_speech_samples:
                        result.speech_ended = True
                        result.speech_audio = bytes(self._speech_buffer)
                    # Reset
                    self._is_speech = False
                    self._speech_buffer = bytearray()
                    self._speech_samples = 0
                    self._silence_samples = 0

        return result
