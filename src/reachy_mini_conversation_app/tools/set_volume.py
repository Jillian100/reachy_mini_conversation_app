"""Volume control tool for Reachy Mini."""

import logging
import re
import subprocess
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _get_current_volume() -> int:
    """Read current PCM volume percentage from amixer."""
    try:
        result = subprocess.run(
            ["amixer", "sget", "PCM"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"\[(\d+)%\]", result.stdout)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return -1


class SetVolume(Tool):
    """Set the robot speaker volume."""

    name = "set_volume"
    description = (
        "Set or adjust the speaker volume. "
        "Use 'absolute' to set a specific level (0-100). "
        "Use 'relative' to increase/decrease by a delta (e.g. +15 or -10). "
        "When user says 'louder/大聲一點', use relative with a positive delta like +15. "
        "When user says 'quieter/小聲一點', use relative with a negative delta like -15."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "volume": {
                "type": "integer",
                "description": "Absolute volume level (0-100). Use this OR delta, not both.",
            },
            "delta": {
                "type": "integer",
                "description": "Relative volume change (e.g. +15 to increase, -10 to decrease). Use this for '大聲一點'/'小聲一點'.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Set volume via amixer, supporting absolute or relative adjustment."""
        current = _get_current_volume()
        delta = kwargs.get("delta")
        volume = kwargs.get("volume")

        if delta is not None:
            # Relative adjustment
            if current < 0:
                current = 70  # fallback if can't read
            target = max(0, min(100, current + int(delta)))
        elif volume is not None:
            target = max(0, min(100, int(volume)))
        else:
            return {"error": "Provide 'volume' (absolute) or 'delta' (relative)"}

        logger.info("Tool call: set_volume current=%d%% → target=%d%%", current, target)

        try:
            result = subprocess.run(
                ["amixer", "sset", "PCM", f"{target}%"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return {"status": "ok", "previous": current, "volume": target}
            return {"error": f"amixer failed: {result.stderr.strip()}"}
        except Exception as e:
            logger.exception("Failed to set volume")
            return {"error": f"Failed to set volume: {e!s}"}
