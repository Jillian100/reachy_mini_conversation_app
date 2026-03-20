"""Dance and emotion moves for the movement queue system.

This module implements dance moves and emotions as Move objects that can be queued
and executed sequentially by the MovementManager.
"""

from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini_dances_library.dance_move import DanceMove


logger = logging.getLogger(__name__)


class DanceQueueMove(Move):  # type: ignore
    """Wrapper for dance moves to work with the movement queue system."""

    def __init__(self, move_name: str):
        """Initialize a DanceQueueMove."""
        self.dance_move = DanceMove(move_name)
        self.move_name = move_name

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.dance_move.duration)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate dance move at time t."""
        try:
            # Get the pose from the dance move
            head_pose, antennas, body_yaw = self.dance_move.evaluate(t)

            # Convert to numpy array if antennas is tuple and return in official Move format
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating dance move '{self.move_name}' at t={t}: {e}")
            # Return neutral pose on error
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)


class EmotionQueueMove(Move):  # type: ignore
    """Wrapper for emotion moves to work with the movement queue system."""

    def __init__(self, emotion_name: str, recorded_moves: RecordedMoves):
        """Initialize an EmotionQueueMove."""
        self.emotion_move = recorded_moves.get(emotion_name)
        self.emotion_name = emotion_name

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float(self.emotion_move.duration)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate emotion move at time t."""
        try:
            # Get the pose from the emotion move
            head_pose, antennas, body_yaw = self.emotion_move.evaluate(t)

            # Convert to numpy array if antennas is tuple and return in official Move format
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]])

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating emotion '{self.emotion_name}' at t={t}: {e}")
            # Return neutral pose on error
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0)


class GotoQueueMove(Move):  # type: ignore
    """Smooth goto move using SDK interpolation curves.

    Supports minjerk (default), ease_in_out, cartoon (overshoot), and linear.
    Minjerk produces natural acceleration/deceleration (the 'gravity groove').
    Cartoon adds playful overshoot — good for expressive gestures.
    """

    def __init__(
        self,
        target_head_pose: NDArray[np.float32],
        start_head_pose: NDArray[np.float32] | None = None,
        target_antennas: Tuple[float, float] = (0, 0),
        start_antennas: Tuple[float, float] | None = None,
        target_body_yaw: float = 0,
        start_body_yaw: float | None = None,
        duration: float = 1.0,
        interpolation: str = "minjerk",
    ):
        """Initialize a GotoQueueMove.

        Args:
            interpolation: "minjerk" | "ease_in_out" | "cartoon" | "linear"
        """
        self._duration = duration
        self.target_head_pose = target_head_pose
        self.start_head_pose = start_head_pose
        self.target_antennas = target_antennas
        self.start_antennas = start_antennas or (0, 0)
        self.target_body_yaw = target_body_yaw
        self.start_body_yaw = start_body_yaw or 0
        self.interpolation = interpolation

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return self._duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate goto move at time t with smooth interpolation."""
        try:
            from reachy_mini.utils import create_head_pose
            from reachy_mini.utils.interpolation import (
                InterpolationTechnique,
                linear_pose_interpolation,
                time_trajectory,
            )

            # Normalize time to [0, 1]
            t_norm = max(0.0, min(1.0, t / self.duration))

            # Apply easing curve — this is what makes the motion feel alive
            interp_map = {
                "minjerk": InterpolationTechnique.MIN_JERK,
                "ease": InterpolationTechnique.EASE_IN_OUT,
                "ease_in_out": InterpolationTechnique.EASE_IN_OUT,
                "cartoon": InterpolationTechnique.CARTOON,
                "linear": InterpolationTechnique.LINEAR,
            }
            technique = interp_map.get(self.interpolation, InterpolationTechnique.MIN_JERK)
            s = time_trajectory(t_norm, technique)

            # Use start pose if available, otherwise neutral
            if self.start_head_pose is not None:
                start_pose = self.start_head_pose
            else:
                start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

            # Interpolate head pose (SLERP rotation + linear translation, driven by eased time)
            head_pose = linear_pose_interpolation(start_pose, self.target_head_pose, s)

            # Interpolate antennas with same easing
            antennas = np.array(
                [
                    self.start_antennas[0] + (self.target_antennas[0] - self.start_antennas[0]) * s,
                    self.start_antennas[1] + (self.target_antennas[1] - self.start_antennas[1]) * s,
                ],
                dtype=np.float64,
            )

            # Interpolate body yaw with same easing
            body_yaw = self.start_body_yaw + (self.target_body_yaw - self.start_body_yaw) * s

            return (head_pose, antennas, body_yaw)

        except Exception as e:
            logger.error(f"Error evaluating goto move at t={t}: {e}")
            target_head_pose_f64 = self.target_head_pose.astype(np.float64)
            target_antennas_array = np.array([self.target_antennas[0], self.target_antennas[1]], dtype=np.float64)
            return (target_head_pose_f64, target_antennas_array, self.target_body_yaw)
