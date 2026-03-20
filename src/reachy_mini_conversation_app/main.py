"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
    log_connection_troubleshooting,
)

# --- MONKEY PATCH START ---
# On macOS: patch zenoh.open to always connect directly to the WiFi robot IP.
# This bypasses multicast discovery which is unreliable on home/studio networks.
if sys.platform == "darwin":
    import json as _json
    import zenoh as _zenoh

    _ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.0.72")
    _original_zenoh_open = _zenoh.open

    def _direct_connect_open(config):
        """Force direct TCP connection to robot instead of multicast discovery."""
        _direct_cfg = _zenoh.Config.from_json5(_json.dumps({
            "mode": "client",
            "connect": {"endpoints": [f"tcp/{_ROBOT_IP}:7447"]},
        }))
        print(f"[MonkeyPatch] zenoh.open → tcp/{_ROBOT_IP}:7447")
        return _original_zenoh_open(_direct_cfg)

    _zenoh.open = _direct_connect_open

    # Patch ReachyMini.__init__ to force no_media on macOS (GStreamer not available)
    # After init, silence the media_manager logger which spams at WARNING level
    import logging as _logging
    from reachy_mini import ReachyMini as _ReachyMini
    _original_init = _ReachyMini.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["media_backend"] = "no_media"
        print("[MonkeyPatch] media_backend=no_media (macOS Gradio mode)")
        _original_init(self, *args, **kwargs)
        # SDK sets logger level inside __init__; override it after
        _logging.getLogger("reachy_mini.media.media_manager").setLevel(_logging.ERROR)

    _ReachyMini.__init__ = _patched_init

    # Also patch ReachyMiniApp to skip robot-side GStreamer media (redundant safety)
    from reachy_mini.apps.app import ReachyMiniApp as _ReachyMiniApp
    _original_wrapped_run = _ReachyMiniApp.wrapped_run

    def _patched_wrapped_run(self):
        self.media_backend = "no_media"
        _original_wrapped_run(self)

    _ReachyMiniApp.wrapped_run = _patched_wrapped_run
# --- MONKEY PATCH END ---


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler

    # Override head_tracker from env var if not set via CLI
    # (config.py has loaded .env by now via the imports above)
    if args.head_tracker is None:
        env_tracker = os.environ.get("REACHY_HEAD_TRACKER")
        if env_tracker in ("yolo", "mediapipe"):
            args.head_tracker = env_tracker

    # Backend selection: "openai" (default), "gemini", or "claude"
    # [MODIFIED] Default to "gemini" for Amelie
    conversation_backend = os.environ.get("CONVERSATION_BACKEND", "gemini").lower()
    if conversation_backend == "gemini":
        from reachy_mini_conversation_app.gemini_live_handler import GeminiLiveHandler as ConversationHandler
    elif conversation_backend == "claude":
        from reachy_mini_conversation_app.claude_pipeline_handler import ClaudePipelineHandler as ConversationHandler
    else:
        from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler as ConversationHandler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")
    logger.info(f"Head tracker: {args.head_tracker}")
    logger.info(f"Backend selected: {conversation_backend.upper()}")

    if args.no_camera and args.head_tracker is not None:
        logger.warning(
            "Head tracking disabled: --no-camera flag is set. "
            "Remove --no-camera to enable head tracking."
        )

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(
                "Connection timeout: Failed to connect to Reachy Mini daemon. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(
                "Connection failed: Unable to establish connection to Reachy Mini. "
                f"Details: {e}"
            )
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(
                f"Unexpected error during robot initialization: {type(e).__name__}: {e}"
            )
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    # Check if running in simulation mode without --gradio
    if robot.client.get_status()["simulation_enabled"] and not args.gradio:
        logger.error(
            "Simulation mode requires Gradio interface. Please use --gradio flag when running in simulation mode."
        )
        robot.client.disconnect()
        sys.exit(1)

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = ConversationHandler(deps, gradio_mode=args.gradio, instance_path=instance_path)
    logger.info("Using conversation backend: %s (%s)", conversation_backend, ConversationHandler.__name__)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        # Determine API Key label and default value based on backend
        if conversation_backend == "gemini":
            key_label = "Gemini API Key"
            default_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif conversation_backend == "claude":
             key_label = "Anthropic API Key"
             default_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            key_label = "OPENAI API Key"
            default_key = os.environ.get("OPENAI_API_KEY")

        api_key_textbox = gr.Textbox(
            label=key_label,
            type="password",
            value=default_key if (not get_space() and default_key) else "",

        )

        from reachy_mini_conversation_app.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[
                chatbot,
                api_key_textbox,
                *personality_ui.additional_inputs_ordered(),
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()

        # is_wireless = reachy_mini.client.get_status()["wireless_version"]
        # args.head_tracker = None if is_wireless else "mediapipe"

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
