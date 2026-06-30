# Audio Duplex Board Prototype

This folder is an independent prototype app for the O5 `display_object_on_board`
demo. It intentionally stays outside the existing gateway / worker / static page
flow so the board interaction can be developed and deleted or migrated safely.

## Scope

Current MVP:

- Run browser-side audio file streaming through `UnifiedProcessor.fc_duplex`.
- Keep static `O5DuplexTrainingData` case replay as a debug endpoint.
- Convert FC replay output into board-facing events.
- Render spoken / think / tool_call summaries and board cards in a standalone page.
- Use deterministic placeholder image cards while the real search backend is being integrated.

Non-goals:

- Do not modify the existing `static/audio-duplex/` page.
- Do not register into gateway / worker pool / homepage navigation.
- Do not put board business logic into `MiniCPMO45/`.
- Do not solve model quality issues; checkpoint can be replaced by CLI args.

## Run

```bash
PYTHONPATH=. .venv/base/bin/python -m audio_duplex_board.run_server \
  --host 0.0.0.0 \
  --port 18080 \
  --model-path /user/weihongliang/autoshow_omni/models/MiniCPM-o-4_5 \
  --pt-path /path/to/minicpm-v_100.pt \
  --sdk-src /user/sunweiyue/lib/swy-dev/omni_agent_research/minicpm_o5_sdk/src
```

Open:

```text
http://localhost:18080/
```

The page supports two prototype flows:

- `File Streaming Replay`: browser decodes an audio file, splits it into 1s
  chunks, sends chunks through `/ws/session`, and plays the user file locally.
- `TrainingData case path`: blocking debug replay through `/api/replay-case`.

## CLI Replay

```bash
PYTHONPATH=. .venv/base/bin/python -m audio_duplex_board.scripts.run_replay_case \
  --case-path /path/to/case.json \
  --output-dir audio_duplex_board/runs/local_case
```

Outputs:

- `events.jsonl`
- `summary.json`

## API

WebSocket:

```text
ws://<host>:<port>/ws/session
```

Messages:

```json
{ "type": "prepare", "payload": { "system_prompt": "...", "tools": [...] } }
{ "type": "audio_chunk", "payload": { "audio_base64": "...", "sample_rate": 16000 } }
{ "type": "finish", "payload": { "reason": "file_replay_finished" } }
```

Debug HTTP endpoint:

```text
POST /api/replay-case
```

## Layout

```text
audio_duplex_board/
├── run_server.py
├── config.py
├── schemas.py
├── session.py
├── replay.py
├── events.py
├── tools/display_object_on_board/
├── web/
├── scripts/
└── runs/
```

`runs/` is ignored by git.
