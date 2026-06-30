"""Configuration for the independent Audio Duplex Board prototype.

This module keeps prototype runtime settings separate from the main Demo
`config.json`, so experiments here do not affect the existing gateway / worker
flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioDuplexBoardConfig:
    """Runtime configuration for the standalone board prototype.

    Args:
        model_path: Local HuggingFace-style MiniCPM-o 4.5 base model directory.
        pt_path: Optional fine-tuned checkpoint overlay path.
        sdk_src: Optional local MiniCPM-O5 SDK source path to prepend to
            `sys.path` before loading training data.
        case_folder: Optional default folder containing TrainingData JSON cases.
        host: HTTP server host.
        port: HTTP server port.
        max_board_cards: Maximum cards kept in the frontend board state.
        image_search_timeout_sec: Timeout budget for image search service.
        use_mock_view: Whether to use a GPU-free mock FcDuplexView.
        mock_energy_threshold: RMS threshold that triggers mock tool calls.
    """

    model_path: str
    pt_path: str | None = None
    sdk_src: str | None = None
    case_folder: str | None = None
    host: str = "127.0.0.1"
    port: int = 18080
    max_board_cards: int = 10
    image_search_timeout_sec: float = 3.0
    use_mock_view: bool = False
    mock_energy_threshold: float = 0.012

    @property
    def case_folder_path(self) -> Path | None:
        """Return `case_folder` as `Path` when configured."""

        return Path(self.case_folder) if self.case_folder else None


DEFAULT_MODEL_PATH = "/user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5"
DEFAULT_SDK_SRC = "/user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_sdk/src"
DEFAULT_CASE_FOLDER = (
    "/user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_training/"
    "experiments/overfit_midtrain_v2_100/runs/subsets/overfit100_seed0/"
    "delivery_train_data"
)


def make_default_config() -> AudioDuplexBoardConfig:
    """Build the default local development config.

    Returns:
        Default config pointing at the known local base model, SDK source, and
        overfit100 case folder. `pt_path` is intentionally left unset and should
        be supplied by CLI for each experiment.
    """

    return AudioDuplexBoardConfig(
        model_path=DEFAULT_MODEL_PATH,
        sdk_src=DEFAULT_SDK_SRC,
        case_folder=DEFAULT_CASE_FOLDER,
    )
