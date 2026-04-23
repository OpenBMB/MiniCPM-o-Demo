#!/usr/bin/env python3
"""验证 Python -> C++ sampling 参数透传是否生效。

直接打 llama-server 的 /v1/stream/update_session_config，传非默认 sampling 值，
并校验 ack 中 sampling 块的回显与请求一致。

用法:
    python scripts/verify_cpp_sampling_passthrough.py [--cpp-port 19072]

预期:
    [PASS] all 4 sampling fields round-tripped
"""
from __future__ import annotations

import argparse
import json
import sys

import urllib.request


def post_json(url: str, body: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-port", type=int, default=19072)
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.cpp_port}/v1/stream/update_session_config"

    # 故意取非默认值，便于和 C++ 默认 (1.0 / 3 / 26 / 0.8) 区分
    targets = {
        "listen_prob_scale": 1.7,
        "force_listen_count": 7,
        "max_new_speak_tokens_per_chunk": 64,
        "tts_temperature": 0.55,
    }

    body = {
        "media_type": 2,
        "duplex_mode": True,
        **targets,
    }

    ack = post_json(url, body)
    sampling_ack = ack.get("sampling")
    if not sampling_ack:
        print("[FAIL] response has no `sampling` block:")
        print(json.dumps(ack, indent=2, ensure_ascii=False))
        return 1

    print("[ack.sampling] " + json.dumps(sampling_ack, ensure_ascii=False))

    failures: list[str] = []
    for key, want in targets.items():
        got = sampling_ack.get(key)
        if got is None:
            failures.append(f"{key}: missing in ack")
            continue
        if isinstance(want, float):
            if abs(float(got) - want) > 1e-3:
                failures.append(f"{key}: got {got!r}, want {want!r}")
        else:
            if int(got) != int(want):
                failures.append(f"{key}: got {got!r}, want {want!r}")

    if failures:
        print("[FAIL] mismatches:")
        for f in failures:
            print(" -", f)
        return 1

    print(f"[PASS] all {len(targets)} sampling fields round-tripped via "
          f"/v1/stream/update_session_config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
