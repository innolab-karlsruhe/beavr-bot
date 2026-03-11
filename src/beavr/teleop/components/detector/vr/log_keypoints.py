import atexit
import json
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class KeypointLogger:
    """
    Handles automatic logging of keypoint and coordinate frame data.

    Features:
    - Accumulates data in memory during runtime
    - Auto-saves at configurable intervals
    - Handles signals (Ctrl+C, SIGTERM) to save data before exit
    - Registers cleanup handlers for normal exit
    """

    def __init__(
        self,
        hand_side: str,
        log_dir: str = "data/keypoint_logs",
        auto_save_interval: int = 100,
        moving_average_limit: int = 5,
    ):
        """
        Initialize the keypoint logger.

        Args:
            hand_side: 'right' or 'left' to identify which hand data is being logged
            log_dir: Directory to save log files (default: "data/keypoint_logs")
            auto_save_interval: Number of frames between auto-saves (default: 100)
            moving_average_limit: Moving average limit used in processing (for metadata)
        """
        self.hand_side = hand_side
        self.log_dir = Path(log_dir)
        self.auto_save_interval = auto_save_interval
        self.moving_average_limit = moving_average_limit

        self.frame_counter = 0
        self.logged_data = []
        self.save_counter = 0

        self._setup_logging()
        self._register_signal_handlers()
        atexit.register(self.save_data)

    def _setup_logging(self):
        """Initialize the logging directory and file path."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_base = self.log_dir / f"keypoints_{self.hand_side}_{timestamp}"
        self.log_file = str(self.log_file_base) + ".json"

        logger.info(f"Logging enabled for {self.hand_side} hand. Data will be saved to: {self.log_file}")

    def _register_signal_handlers(self):
        """Register signal handlers to save data on program termination."""

        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}. Saving logged data before exit...")
            self.save_data()
            # Re-raise to allow proper shutdown
            signal.signal(sig, signal.SIG_DFL)
            signal.raise_signal(sig)

        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    def save_data(self):
        """Save the accumulated logged data to a JSON file."""
        if len(self.logged_data) == 0:
            return

        try:
            self.save_counter += 1
            if self.save_counter == 1:
                log_file = str(self.log_file_base) + ".json"
            else:
                log_file = str(self.log_file_base) + f"_{self.save_counter}.json"

            serializable_data = {
                "hand_side": self.hand_side,
                "total_frames": len(self.logged_data),
                "frame_offset": (self.save_counter - 1) * self.auto_save_interval,
                "frames": self.logged_data.copy(),
                "metadata": {
                    "save_timestamp": datetime.now().isoformat(),
                    "moving_average_limit": self.moving_average_limit,
                    "save_counter": self.save_counter,
                },
            }

            with open(log_file, "w") as f:
                json.dump(serializable_data, f, indent=2)

            logger.info(f"Saved {len(self.logged_data)} frames to {log_file}")
            self.logged_data.clear()
        except Exception as e:
            logger.error(f"Error saving logged data: {e}")

    def log_frame(self, keypoints, coordinate_frame):
        """
        Accumulate frame data for logging and trigger auto-saves.

        Args:
            keypoints: Hand keypoints array (N, 3)
            coordinate_frame: Coordinate frame [origin, x_vec, y_vec, z_vec]
        """
        # Convert numpy arrays to lists for JSON serialization
        frame_data = {
            "frame_id": self.frame_counter,
            "timestamp": time.time(),
            "keypoints": keypoints.tolist() if isinstance(keypoints, np.ndarray) else keypoints,
            "coordinate_frame": {
                "origin": coordinate_frame[0].tolist()
                if isinstance(coordinate_frame[0], np.ndarray)
                else coordinate_frame[0],
                "x_vector": coordinate_frame[1].tolist()
                if isinstance(coordinate_frame[1], np.ndarray)
                else coordinate_frame[1],
                "y_vector": coordinate_frame[2].tolist()
                if isinstance(coordinate_frame[2], np.ndarray)
                else coordinate_frame[2],
                "z_vector": coordinate_frame[3].tolist()
                if isinstance(coordinate_frame[3], np.ndarray)
                else coordinate_frame[3],
            },
        }

        self.logged_data.append(frame_data)
        self.frame_counter += 1

        # Auto-save at specified intervals
        if self.frame_counter % self.auto_save_interval == 0:
            self.save_data()
            logger.debug(f"Auto-saved at frame {self.frame_counter}")
