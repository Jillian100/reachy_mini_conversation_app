"""Gemini Live API handler for real-time audio/video conversations with Reachy Mini.

Drop-in replacement for OpenaiRealtimeHandler. Uses Google's Gemini Live API
for bidirectional audio streaming with native speech-to-speech capability.

Audio format: raw PCM, 16-bit little-endian, mono
Input (to Gemini):  16 kHz
Output (from Gemini): 24 kHz

Reference implementation: gamepop/reachy-mini-gemini (proven on Reachy Mini hardware)
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
from typing import Any, Dict, Final, List, Optional, Tuple
from datetime import datetime

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample as scipy_resample

from google import genai
from google.genai import types

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_voice
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

# Gemini Live API audio parameters
GEMINI_INPUT_SAMPLE_RATE: Final[int] = 16000
GEMINI_OUTPUT_SAMPLE_RATE: Final[int] = 24000

# Default model — Gemini 2.5 Flash with native audio (Live API)
# Note: Gemini 3 Flash does NOT support Live API as of 2026-02
DEFAULT_GEMINI_MODEL: Final[str] = "models/gemini-2.5-flash-native-audio-preview-12-2025"

# Volume gain for Gemini audio output (Gemini tends to output quieter audio)
VOLUME_GAIN: Final[float] = float(os.environ.get("GEMINI_VOLUME_GAIN", "3.0"))

# OpenAI voice → Gemini voice mapping
_OPENAI_TO_GEMINI_VOICE: Dict[str, str] = {
    "coral": "Kore",
    "alloy": "Aoede",
    "aria": "Leda",
    "ballad": "Fenrir",
    "sage": "Charon",
    "verse": "Puck",
    "cedar": "Zephyr",
}

GEMINI_VOICES: Final[List[str]] = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Aoede",
    "Leda", "Orus", "Autonoe", "Erinome", "Schedar", "Sadachbia",
    "Algieba", "Achernar", "Zubenelgenubi", "Callirrhoe", "Despina",
    "Alnilam", "Vindemiatrix", "Umbriel", "Laomedeia", "Achird",
    "Enceladus", "Algenib", "Gacrux", "Sadaltager", "Iapetus",
    "Rasalgethi", "Pulcherrima", "Sulafat",
]


def _convert_tools_to_gemini(tool_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-format tool specs to Gemini function_declarations.

    OpenAI format:  {"type": "function", "name": ..., "description": ..., "parameters": {...}}
    Gemini format:  {"function_declarations": [{"name": ..., "description": ..., "parameters": {...}}]}
    """
    declarations: List[Dict[str, Any]] = []
    for spec in tool_specs:
        decl: Dict[str, Any] = {
            "name": spec["name"],
            "description": spec.get("description", ""),
        }
        params = spec.get("parameters")
        if params and isinstance(params, dict) and params.get("properties"):
            decl["parameters"] = params
        declarations.append(decl)
    return [{"function_declarations": declarations}]


class GeminiLiveHandler(AsyncStreamHandler):
    """Gemini Live API handler implementing fastrtc AsyncStreamHandler.

    Compatible with both Gradio mode and headless LocalStream mode.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=GEMINI_INPUT_SAMPLE_RATE,  # Output at 16kHz (robot native)
            input_sample_rate=GEMINI_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Session state
        self.session: Any = None
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()

        # Timing
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False

        # Lifecycle
        self._shutdown_requested = False
        self._reconnect_requested = False
        self._connected_event = asyncio.Event()
        self._receive_task: Optional[asyncio.Task[None]] = None

        # Echo suppression: mute mic input while robot is speaking
        self._model_speaking = False
        self._last_audio_data_time: float = 0.0

        # Audio buffer for resampling (accumulate small Gemini chunks before output)
        self._audio_buffer = bytearray()
        # Output at robot's native 16kHz to avoid per-chunk resampling artifacts
        self._output_sr = GEMINI_INPUT_SAMPLE_RATE  # 16000

    def copy(self) -> GeminiLiveHandler:
        """Create a copy of the handler (required by fastrtc for Gradio mode)."""
        return GeminiLiveHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------ #
    #  API key resolution
    # ------------------------------------------------------------------ #
    def _resolve_api_key(self) -> str:
        """Resolve Gemini API key from environment."""
        key = (
            os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or getattr(config, "GEMINI_API_KEY", None)
            or ""
        )
        return key.strip()

    # ------------------------------------------------------------------ #
    #  Voice resolution (maps OpenAI voice names to Gemini)
    # ------------------------------------------------------------------ #
    def _resolve_voice(self) -> str:
        """Get the voice name for Gemini, mapping OpenAI names if needed."""
        voice_str = get_session_voice(default="cedar")
        mapped = _OPENAI_TO_GEMINI_VOICE.get(voice_str.lower())
        if mapped:
            return mapped
        # If the voice name is already a valid Gemini voice, use it directly
        if voice_str in GEMINI_VOICES:
            return voice_str
        # Fallback
        return "Kore"

    # ------------------------------------------------------------------ #
    #  Lifecycle: start_up / shutdown
    # ------------------------------------------------------------------ #
    async def start_up(self) -> None:
        """Connect to Gemini Live API with auto-reconnect on session expiry."""
        api_key = self._resolve_api_key()
        if not api_key:
            logger.error(
                "No Gemini API key found. Set GEMINI_API_KEY or GOOGLE_API_KEY in environment."
            )
            return

        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )

        # Outer reconnect loop (handles 15-min session limit)
        max_consecutive_failures = 5
        consecutive_failures = 0
        while not self._shutdown_requested:
            try:
                await self._run_session()
                consecutive_failures = 0  # Reset on successful session
                if self._shutdown_requested:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._shutdown_requested:
                    break
                consecutive_failures += 1
                logger.warning(
                    "Gemini session ended (%s). Failure %d/%d.",
                    e, consecutive_failures, max_consecutive_failures,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Gemini: %d consecutive failures — stopping reconnect. "
                        "Check API key and network.",
                        max_consecutive_failures,
                    )
                    break
                self.session = None
                self._connected_event.clear()
                # Exponential backoff: 2s, 4s, 8s, 16s...
                backoff = min(2 ** consecutive_failures, 30)
                await asyncio.sleep(backoff)

    async def _run_session(self) -> None:
        """Establish and manage a single Gemini Live session."""
        instructions = get_session_instructions()
        voice = self._resolve_voice()
        tool_specs = get_tool_specs()
        gemini_tools = _convert_tools_to_gemini(tool_specs)

        model = os.environ.get("GEMINI_LIVE_MODEL", DEFAULT_GEMINI_MODEL)

        session_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice,
                    )
                ),
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=instructions)]
            ),
            tools=gemini_tools,
        )

        logger.info(
            "Connecting to Gemini Live: model=%s voice=%s profile=%r",
            model,
            voice,
            getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
        )

        async with self.client.aio.live.connect(model=model, config=session_config) as session:
            self.session = session
            self._connected_event.set()
            self._reconnect_requested = False
            logger.info("Gemini Live session connected")

            try:
                await self._receive_loop()
            finally:
                self.session = None
                self._connected_event.clear()

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_requested = True

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        self.session = None

        # Drain output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ------------------------------------------------------------------ #
    #  Receive loop: audio, tool calls, transcription from Gemini
    # ------------------------------------------------------------------ #
    async def _receive_loop(self) -> None:
        """Continuously receive from Gemini session and dispatch events."""
        while self.session and not self._shutdown_requested and not self._reconnect_requested:
            try:
                async for response in self.session.receive():
                    if self._shutdown_requested or self._reconnect_requested:
                        break

                    # --- Audio data ---
                    if response.data is not None:
                        import time as _time
                        self._model_speaking = True
                        self._last_audio_data_time = _time.time()
                        self.last_activity_time = asyncio.get_event_loop().time()

                        # Feed head wobbler for audio-reactive motion
                        if self.deps.head_wobbler is not None:
                            import base64
                            self.deps.head_wobbler.feed(
                                base64.b64encode(response.data).decode()
                            )

                        # Buffer small chunks, then resample as a batch to avoid
                        # per-chunk resampling artifacts (Gemini sends ~40ms chunks)
                        self._audio_buffer.extend(response.data)

                        # Flush when we have >= 200ms of audio (4800 samples @ 24kHz)
                        min_bytes = 4800 * 2  # 200ms @ 24kHz, 16-bit
                        if len(self._audio_buffer) >= min_bytes:
                            await self._flush_audio_buffer()

                    # --- Text response ---
                    if response.text is not None:
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {"role": "assistant", "content": response.text}
                            )
                        )

                    # --- Tool calls ---
                    if hasattr(response, "tool_call") and response.tool_call:
                        await self._handle_tool_calls(response.tool_call)

                    # --- Server content (transcription, interruption) ---
                    if response.server_content is not None:
                        await self._handle_server_content(response.server_content)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._shutdown_requested:
                    break
                logger.warning("Receive loop error: %s", e)
                # Session likely expired — exit to trigger reconnect
                break

    async def _handle_server_content(self, sc: Any) -> None:
        """Process Gemini server_content for transcription and interruption."""
        # Input transcription (user speech → text)
        if hasattr(sc, "input_transcription") and sc.input_transcription:
            text = getattr(sc.input_transcription, "text", None)
            if text:
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": text})
                )

        # Output transcription (model speech → text)
        if hasattr(sc, "output_transcription") and sc.output_transcription:
            text = getattr(sc.output_transcription, "text", None)
            if text:
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": text})
                )

        # Interruption (user started speaking while model was talking)
        if hasattr(sc, "interrupted") and sc.interrupted:
            self._model_speaking = False
            self._audio_buffer.clear()
            logger.debug("Model interrupted by user speech")
            if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                self._clear_queue()
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()

        # Turn complete — flush remaining audio buffer and re-enable mic
        if hasattr(sc, "turn_complete") and sc.turn_complete:
            if self._audio_buffer:
                await self._flush_audio_buffer()
            self._model_speaking = False
            logger.debug("Gemini turn complete")

    # ------------------------------------------------------------------ #
    #  Audio buffer flush: resample 24kHz → 16kHz in batch
    # ------------------------------------------------------------------ #
    async def _flush_audio_buffer(self) -> None:
        """Resample buffered 24kHz audio to 16kHz and push to output queue."""
        if not self._audio_buffer:
            return

        raw = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        audio_24k = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

        # Apply volume gain
        if VOLUME_GAIN != 1.0:
            audio_24k = audio_24k * VOLUME_GAIN

        # Resample 24kHz → 16kHz in one batch (avoids per-chunk artifacts)
        num_samples_16k = int(len(audio_24k) * self._output_sr / GEMINI_OUTPUT_SAMPLE_RATE)
        audio_16k = scipy_resample(audio_24k, num_samples_16k)

        # Clip and convert back to int16
        audio_out = np.clip(audio_16k, -32768, 32767).astype(np.int16)

        await self.output_queue.put((self._output_sr, audio_out))

    # ------------------------------------------------------------------ #
    #  Tool call handling (uses existing core_tools dispatch)
    # ------------------------------------------------------------------ #
    async def _handle_tool_calls(self, tool_call: Any) -> None:
        """Process Gemini tool calls via existing dispatch_tool_call system."""
        function_responses: List[types.FunctionResponse] = []

        for fc in tool_call.function_calls:
            tool_name = fc.name
            args_dict = dict(fc.args) if fc.args else {}
            args_json = json.dumps(args_dict)

            logger.info("Tool call: %s(%s)", tool_name, args_json)

            try:
                tool_result = await dispatch_tool_call(tool_name, args_json, self.deps)
                logger.debug("Tool '%s' result: %s", tool_name, tool_result)
            except Exception as e:
                logger.error("Tool '%s' failed: %s", tool_name, e)
                tool_result = {"error": str(e)}

            # Emit tool result to chatbot UI
            await self.output_queue.put(
                AdditionalOutputs({
                    "role": "assistant",
                    "content": json.dumps(tool_result),
                    "metadata": {"title": f"Used tool {tool_name}", "status": "done"},
                })
            )

            # Camera tool: send captured image to Gemini for visual understanding
            if tool_name == "camera" and "b64_im" in tool_result:
                import base64 as b64mod
                try:
                    img_bytes = b64mod.b64decode(tool_result["b64_im"])
                    if self.session:
                        await self.session.send(
                            input={"data": img_bytes, "mime_type": "image/jpeg"}
                        )
                        logger.info("Sent camera image to Gemini session")
                except Exception as e:
                    logger.warning("Failed to send camera image: %s", e)

                # Emit image to Gradio chatbot
                if self.deps.camera_worker is not None:
                    try:
                        import cv2
                        import gradio as gr
                        np_img = self.deps.camera_worker.get_latest_frame()
                        if np_img is not None:
                            rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                        else:
                            rgb_frame = None
                        await self.output_queue.put(
                            AdditionalOutputs(
                                {"role": "assistant", "content": gr.Image(value=rgb_frame)}
                            )
                        )
                    except Exception:
                        pass

            # Build function response for Gemini
            response_payload = tool_result if isinstance(tool_result, dict) else {"result": str(tool_result)}
            function_responses.append(
                types.FunctionResponse(
                    name=fc.name,
                    id=fc.id,
                    response=response_payload,
                )
            )

        # Send all tool responses back to Gemini
        if self.session and function_responses:
            await self.session.send(
                input=types.LiveClientToolResponse(
                    function_responses=function_responses,
                )
            )

        # Reset head wobbler after tool execution
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    # ------------------------------------------------------------------ #
    #  Audio I/O: receive() and emit()
    # ------------------------------------------------------------------ #
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from mic and send to Gemini.

        Args:
            frame: (sample_rate, audio_data) tuple from microphone.

        """
        if not self.session:
            return

        # Echo suppression: skip mic input while robot is speaking
        # Safety: auto-reset after 3s of no new audio from Gemini
        if self._model_speaking:
            import time as _time
            if _time.time() - self._last_audio_data_time > 3.0:
                self._model_speaking = False
                logger.debug("Echo suppression auto-reset (3s timeout)")
            else:
                return

        input_sample_rate, audio_frame = frame

        # Reshape: ensure 1D mono
        if audio_frame.ndim == 2:
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]
            else:
                audio_frame = audio_frame.ravel()

        # Resample to 16 kHz if needed
        if input_sample_rate != GEMINI_INPUT_SAMPLE_RATE:
            num_samples = int(len(audio_frame) * GEMINI_INPUT_SAMPLE_RATE / input_sample_rate)
            audio_frame = scipy_resample(audio_frame, num_samples)

        # Ensure int16
        audio_frame = audio_to_int16(audio_frame)

        # Send raw PCM bytes to Gemini
        try:
            await self.session.send(
                input={"data": audio_frame.tobytes(), "mime_type": "audio/pcm"}
            )
        except Exception:
            # Session may be closing/reconnecting — drop frame silently
            pass

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio/outputs to speaker and chatbot."""
        # Idle behavior — DISABLED (2026-02-22)
        # Gemini Live native audio ignores system_instruction idle directives.
        # send_idle_signal() sends "express yourself" as user message, which
        # overrides profile instructions and causes unprompted motivational speech.
        # Fix: stop sending idle signals entirely. Robot stays silent when idle.

        return await wait_for_item(self.output_queue)

    # ------------------------------------------------------------------ #
    #  Personality
    # ------------------------------------------------------------------ #
    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality by triggering session reconnect."""
        try:
            from reachy_mini_conversation_app.config import set_custom_profile
            set_custom_profile(profile)
            logger.info("Set profile to %r, requesting reconnect", profile)
            self._reconnect_requested = True
            return "Applied personality. Reconnecting with new instructions..."
        except Exception as e:
            logger.error("apply_personality failed: %s", e)
            return f"Failed to apply personality: {e}"

    # ------------------------------------------------------------------ #
    #  Idle signal
    # ------------------------------------------------------------------ #
    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send idle signal to Gemini to trigger spontaneous behavior."""
        if not self.session:
            return
        self.is_idle_tool_call = True
        elapsed = asyncio.get_event_loop().time() - self.start_time
        dt = datetime.now()
        timestamp = f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"
        msg = (
            f"{timestamp} Idle for {idle_duration:.0f}s. "
            "Feel free to express yourself — dance, show an emotion, look around, or just be yourself!"
        )
        await self.session.send_client_content(
            turns=types.Content(parts=[types.Part(text=msg)]),
            turn_complete=True,
        )

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        elapsed = asyncio.get_event_loop().time() - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Return available Gemini voices."""
        return list(GEMINI_VOICES)
