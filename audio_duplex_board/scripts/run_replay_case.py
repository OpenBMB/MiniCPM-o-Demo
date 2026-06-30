"""Run one TrainingData case through the board prototype and write events."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audio_duplex_board.config import AudioDuplexBoardConfig, make_default_config
from audio_duplex_board.replay import replay_case_path
from audio_duplex_board.session import AudioDuplexBoardSession
from core.processors import UnifiedProcessor


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    defaults = make_default_config()
    parser = argparse.ArgumentParser(description="Replay one case through audio_duplex_board")
    parser.add_argument("--case-path", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=defaults.model_path)
    parser.add_argument("--pt-path", default=None)
    parser.add_argument("--sdk-src", default=defaults.sdk_src)
    return parser.parse_args()


def main() -> None:
    """Run replay and write `events.jsonl` / `summary.json`."""

    args = parse_args()
    if args.sdk_src and args.sdk_src not in sys.path:
        sys.path.insert(0, args.sdk_src)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AudioDuplexBoardConfig(
        model_path=args.model_path,
        pt_path=args.pt_path,
        sdk_src=args.sdk_src,
    )
    processor = UnifiedProcessor(
        model_path=config.model_path,
        pt_path=config.pt_path,
        device="cuda",
        compile=False,
        attn_implementation="sdpa",
    )
    session = AudioDuplexBoardSession(config=config, processor=processor)
    response = replay_case_path(
        session=session,
        case_path=Path(args.case_path),
        data_root=Path(args.data_root) if args.data_root else None,
    )
    with (output_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in response.events:
            handle.write(event.model_dump_json() + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(response.model_dump(), handle, ensure_ascii=False, indent=2)
    print(f"events={output_dir / 'events.jsonl'}")
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
