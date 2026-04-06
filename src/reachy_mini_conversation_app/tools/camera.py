import base64
import asyncio
import logging
import os
from typing import Any, Dict

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Max image dimension (pixels) — prevents 456KB+ base64 from crashing Gemini session
CAMERA_MAX_DIM: int = int(os.environ.get("CAMERA_MAX_DIM", "640"))
CAMERA_JPEG_QUALITY: int = int(os.environ.get("CAMERA_JPEG_QUALITY", "60"))


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Get frame from camera worker buffer (like main_works.py)
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Use vision manager for processing if available
        if deps.vision_manager is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, image_query,
            )
            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        # Guard mode: local-only vision via Ollama Gemma 26B (zero cloud)
        if os.environ.get("VICKY_MODE") == "guard":
            return await self._local_vision(frame, image_query)

        # Resize if too large (prevent Gemini session overflow)
        h, w = frame.shape[:2]
        if max(h, w) > CAMERA_MAX_DIM:
            scale = CAMERA_MAX_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
            logger.info("Camera frame resized: %dx%d → %dx%d",
                        w, h, frame.shape[1], frame.shape[0])

        # Encode as compressed JPEG
        success, buffer = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        b64_encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        kb = len(b64_encoded) / 1024
        logger.info("Camera image: %dx%d, quality=%d, %.1fKB base64",
                     frame.shape[1], frame.shape[0], CAMERA_JPEG_QUALITY, kb)
        return {"b64_im": b64_encoded}

    async def _local_vision(self, frame, question: str) -> Dict[str, Any]:
        """Guard mode: analyze image locally via Ollama Gemma 26B. Zero cloud."""
        import time as _t

        # Resize for local model
        h, w = frame.shape[:2]
        if max(h, w) > CAMERA_MAX_DIM:
            scale = CAMERA_MAX_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        success, buf = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])
        if not success:
            return {"error": "Failed to encode frame"}

        b64_img = base64.b64encode(buf.tobytes()).decode("utf-8")

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        # Strip /v1 suffix if present (Ollama native API doesn't use it)
        ollama_url = ollama_url.replace("/v1", "")
        model = os.environ.get("OLLAMA_VISION_MODEL", "gemma4:26b")

        t0 = _t.monotonic()
        try:
            import urllib.request
            import json

            payload = json.dumps({
                "model": model,
                "prompt": question,
                "images": [b64_img],
                "stream": False,
                "options": {"num_predict": 200},
            }).encode()

            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            def _call():
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read())

            result = await asyncio.to_thread(_call)
            elapsed = _t.monotonic() - t0
            answer = result.get("response", "")
            logger.info("Guard vision (local %s): %.1fs, %d chars",
                        model, elapsed, len(answer))
            return {"image_description": answer, "local_only": True}

        except Exception as e:
            logger.error("Local vision failed: %s", e)
            return {"error": f"Local vision failed: {e}"}
