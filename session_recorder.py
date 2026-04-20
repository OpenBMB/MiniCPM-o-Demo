"""Session 录制模块

自动录制 Worker 端所有模型 I/O 和原始资源，持久化到 data/sessions/<session_id>/ 目录。
支持 Duplex（chunk-based）和 Turn-based（Streaming/Chat）两种录制模式。

音频格式：16-bit PCM WAV（.wav），自描述、通用可播放、存储减半。

核心类：
    - SessionRecorder: 基类，管理目录、文件写入、timeline
    - DuplexSessionRecorder: Duplex 模式，逐 chunk 录制
    - TurnBasedSessionRecorder: Streaming/Chat 模式，逐 turn 录制

使用：
    recorder = DuplexSessionRecorder(
        session_id="adx_m3kf92",
        app_type="audio_duplex",
        worker_id=0,
        config_snapshot={"system_prompt": "...", "ref_audio": "..."},
        data_dir="data",
    )
    recorder.save_user_audio(0, audio_waveform)
    recorder.record_chunk(0, receive_ts_ms=0, result_dict={...}, prefill_ms=92.0)
    recorder.finalize()
"""

import base64
import json
import logging
import os
import shutil
import string
import struct
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor, Future, wait as _futures_wait
from datetime import datetime, timezone
from random import choices
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_io_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="session_io")


def generate_session_id(prefix: str) -> str:
    """生成 session_id: {prefix}_{6位base36随机}

    Args:
        prefix: 应用前缀，如 chat, stm, adx, omni

    Returns:
        如 "adx_m3kf92"
    """
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(choices(alphabet, k=6))
    return f"{prefix}_{suffix}"


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _write_wav(path: str, pcm_float32: np.ndarray, sample_rate: int) -> None:
    """将 Float32 numpy 数组写为 16-bit PCM WAV 文件

    Args:
        path: 输出文件路径
        pcm_float32: 音频数据（float32, mono）
        sample_rate: 采样率（user_audio=16000, ai_audio=24000）
    """
    pcm16 = np.clip(pcm_float32 * 32767, -32768, 32767).astype(np.int16)
    n_channels = 1
    sampwidth = 2
    data_size = len(pcm16) * sampwidth
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH",
                            16, 1, n_channels, sample_rate,
                            sample_rate * n_channels * sampwidth,
                            n_channels * sampwidth, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(pcm16.tobytes())


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_wav_mono(path: str) -> Optional[np.ndarray]:
    """读取 16-bit PCM WAV 文件为 float32 数组，失败返回 None"""
    try:
        with wave.open(path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            pcm16 = np.frombuffer(raw, dtype=np.int16)
            if wf.getnchannels() > 1:
                pcm16 = pcm16[::wf.getnchannels()]
            return pcm16.astype(np.float32) / 32767.0
    except Exception:
        return None


def _resample_linear(pcm: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """线性插值重采样"""
    if from_sr == to_sr or len(pcm) == 0:
        return pcm
    new_len = int(len(pcm) * to_sr / from_sr)
    if new_len == 0:
        return np.array([], dtype=np.float32)
    return np.interp(
        np.linspace(0, len(pcm) - 1, new_len),
        np.arange(len(pcm)),
        pcm,
    ).astype(np.float32)


def _write_stereo_wav(path: str, left: np.ndarray, right: np.ndarray, sample_rate: int) -> None:
    """写入双声道 16-bit PCM WAV（left=左声道, right=右声道）"""
    n = min(len(left), len(right))
    l16 = np.clip(left[:n] * 32767, -32768, 32767).astype(np.int16)
    r16 = np.clip(right[:n] * 32767, -32768, 32767).astype(np.int16)
    stereo = np.empty(n * 2, dtype=np.int16)
    stereo[0::2] = l16
    stereo[1::2] = r16
    data_bytes = stereo.tobytes()
    data_size = len(data_bytes)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH",
                            16, 1, 2, sample_rate,
                            sample_rate * 2 * 2, 2 * 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(data_bytes)


class SessionRecorder:
    """Session 录制基类

    管理 session 目录结构、异步文件写入、meta.json 和 recording.json 的生命周期。
    子类实现具体的 timeline 条目格式（chunk vs turn）。
    """

    def __init__(
        self,
        session_id: str,
        app_type: str,
        worker_id: int,
        config_snapshot: Dict[str, Any],
        data_dir: str = "data",
    ) -> None:
        """
        Args:
            session_id: 全局唯一 session 标识
            app_type: 应用类型 (chat / streaming / audio_duplex / omni_duplex)
            worker_id: 处理该 session 的 Worker GPU ID
            config_snapshot: 模型配置快照 (system_prompt, ref_audio, length_penalty, ...)
            data_dir: 数据根目录（相对于 minicpmo45_service/）
        """
        self.session_id = session_id
        self.app_type = app_type
        self.worker_id = worker_id
        self.config_snapshot = config_snapshot

        base = os.path.join(os.path.dirname(__file__), data_dir, "sessions", session_id)
        self.session_dir = base
        self._start_ts = datetime.now(timezone.utc)

        os.makedirs(os.path.join(base, "user_audio"), exist_ok=True)
        os.makedirs(os.path.join(base, "user_frames"), exist_ok=True)
        os.makedirs(os.path.join(base, "ai_audio"), exist_ok=True)
        os.makedirs(os.path.join(base, "user_images"), exist_ok=True)
        os.makedirs(os.path.join(base, "user_videos"), exist_ok=True)

        meta = {
            "session_id": session_id,
            "app_type": app_type,
            "created_at": self._start_ts.isoformat(),
            "ended_at": None,
            "duration_s": None,
            "worker_id": worker_id,
            "title": f"对话 {self._start_ts.strftime('%m-%d %H:%M')}",
            "config": config_snapshot,
        }
        _write_json(os.path.join(base, "meta.json"), meta)

        self._finalized = False
        self._pending_io: List[Future] = []
        logger.info(f"[SessionRecorder] created: {session_id} ({app_type}) → {base}")

    # ========== 文件保存（异步提交到线程池） ==========

    def save_user_audio(self, chunk_index: int, pcm_float32: np.ndarray) -> str:
        """保存用户音频 chunk 为 16-bit PCM WAV (16kHz mono)

        Returns:
            相对路径，如 "user_audio/000.wav"
        """
        rel = f"user_audio/{chunk_index:03d}.wav"
        path = os.path.join(self.session_dir, rel)
        self._pending_io.append(_io_pool.submit(_write_wav, path, pcm_float32, 16000))
        return rel

    def save_user_frame(self, chunk_index: int, jpeg_bytes: bytes) -> str:
        """保存用户视频帧（JPEG）

        Returns:
            相对路径，如 "user_frames/000.jpg"
        """
        rel = f"user_frames/{chunk_index:03d}.jpg"
        path = os.path.join(self.session_dir, rel)
        self._pending_io.append(_io_pool.submit(_write_bytes, path, jpeg_bytes))
        return rel

    def save_ai_audio(self, turn_index: int, chunk_index: int, pcm_float32: np.ndarray) -> str:
        """保存 AI 生成的音频为 16-bit PCM WAV (24kHz mono)

        Returns:
            相对路径，如 "ai_audio/turn_0_chunk_000.wav"
        """
        rel = f"ai_audio/turn_{turn_index}_chunk_{chunk_index:03d}.wav"
        path = os.path.join(self.session_dir, rel)
        self._pending_io.append(_io_pool.submit(_write_wav, path, pcm_float32, 24000))
        return rel

    def save_ai_audio_turn(self, turn_index: int, pcm_float32: np.ndarray) -> str:
        """保存整轮 AI 音频为 16-bit PCM WAV (24kHz mono)

        Returns:
            相对路径，如 "ai_audio/turn_0.wav"
        """
        rel = f"ai_audio/turn_{turn_index}.wav"
        path = os.path.join(self.session_dir, rel)
        self._pending_io.append(_io_pool.submit(_write_wav, path, pcm_float32, 24000))
        return rel

    def save_user_image(self, image_index: int, image_data: bytes) -> str:
        """保存用户上传的图片

        Returns:
            相对路径，如 "user_images/img_0.jpg"
        """
        rel = f"user_images/img_{image_index}.jpg"
        path = os.path.join(self.session_dir, rel)
        self._pending_io.append(_io_pool.submit(_write_bytes, path, image_data))
        return rel

    def save_user_video(self, video_index: int, video_data: bytes, ext: str = "mp4") -> str:
        """保存用户上传的视频

        Returns:
            相对路径，如 "user_videos/vid_0.mp4"
        """
        rel = f"user_videos/vid_{video_index}.{ext}"
        path = os.path.join(self.session_dir, rel)
        self._pending_io.append(_io_pool.submit(_write_bytes, path, video_data))
        return rel

    def update_config(self, extra: Dict[str, Any]) -> None:
        """追加配置信息到 config_snapshot（用于延迟获取的参数如 system_prompt）"""
        self.config_snapshot.update(extra)

    def _build_recording_json(self) -> Dict[str, Any]:
        """子类实现：构建 recording.json 内容"""
        raise NotImplementedError

    def _wait_pending_io(self) -> None:
        """等待所有挂起的异步 I/O 完成"""
        if self._pending_io:
            _futures_wait(self._pending_io)
            self._pending_io.clear()

    def _finalize_hook(self, recording: Dict[str, Any]) -> None:
        """子类钩子：在 recording.json 写入前执行后处理（如拼接回放文件）"""
        pass

    def finalize(self) -> None:
        """结束录制：等待 I/O、执行后处理钩子、flush recording.json + meta.json"""
        if self._finalized:
            return
        self._finalized = True

        end_ts = datetime.now(timezone.utc)
        duration_s = (end_ts - self._start_ts).total_seconds()

        meta_path = os.path.join(self.session_dir, "meta.json")
        recording = self._build_recording_json()

        def _flush() -> None:
            self._wait_pending_io()
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            meta["ended_at"] = end_ts.isoformat()
            meta["duration_s"] = round(duration_s, 1)
            _write_json(meta_path, meta)
            self._finalize_hook(recording)
            _write_json(os.path.join(self.session_dir, "recording.json"), recording)

        _io_pool.submit(_flush)
        logger.info(
            f"[SessionRecorder] finalized: {self.session_id} "
            f"({duration_s:.1f}s, dir={self.session_dir})"
        )


class DuplexSessionRecorder(SessionRecorder):
    """Duplex 模式录制器

    逐 audio_chunk 记录 timeline，每个 chunk 一条记录。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._chunks: List[Dict[str, Any]] = []
        self._turn_index: int = 0
        self._speak_chunk_in_turn: int = 0
        # AI 音频独立时间轴：duplex 下 C++ 产出的 wav 通过 _wav_poll_loop 异步到达，
        # 与用户 audio_chunk 并不是 1:1，所以单独按接收时间戳记录，stitch 时再对齐右声道。
        self._ai_audio_timeline: List[Dict[str, Any]] = []
        self._ai_audio_seq: int = 0

    def record_chunk(
        self,
        index: int,
        receive_ts_ms: float,
        result_dict: Dict[str, Any],
        prefill_ms: float,
        user_audio_rel: Optional[str] = None,
        user_frame_rel: Optional[str] = None,
        ai_audio_rel: Optional[str] = None,
        ai_audio_samples: int = 0,
    ) -> None:
        """记录一个 chunk 到 timeline

        Args:
            index: chunk 序号
            receive_ts_ms: 相对于 session 开始的接收时间 (ms)
            result_dict: DuplexGenerateResult.model_dump() 的内容
            prefill_ms: prefill 耗时
            user_audio_rel: 用户音频相对路径
            user_frame_rel: 用户帧相对路径
            ai_audio_rel: AI 音频相对路径
            ai_audio_samples: AI 音频采样点数（用于计算时长）
        """
        is_listen = result_dict.get("is_listen", True)
        mode = "LISTEN" if is_listen else "SPEAK"

        timing: Dict[str, Any] = {"prefill_ms": round(prefill_ms, 1)}
        for key in ("cost_llm_ms", "cost_tts_prep_ms", "cost_tts_ms",
                     "cost_token2wav_ms", "cost_all_ms", "wall_clock_ms"):
            val = result_dict.get(key)
            if val is not None:
                timing[key.replace("cost_", "")] = round(val, 1) if isinstance(val, float) else val
        for key in ("n_tokens", "n_tts_tokens", "kv_cache_length",
                     "vision_slices", "vision_tokens"):
            val = result_dict.get(key)
            if val is not None:
                timing[key] = val

        entry: Dict[str, Any] = {
            "index": index,
            "receive_ts_ms": round(receive_ts_ms, 1),
        }
        if user_audio_rel:
            entry["user_audio"] = user_audio_rel
        if user_frame_rel:
            entry["user_frame"] = user_frame_rel

        result_entry: Dict[str, Any] = {"mode": mode, "timing": timing}
        if not is_listen:
            text = result_dict.get("text", "")
            if text:
                result_entry["text"] = text
            if ai_audio_rel:
                result_entry["ai_audio"] = ai_audio_rel
                if ai_audio_samples > 0:
                    result_entry["ai_audio_duration_ms"] = round(ai_audio_samples / 24000 * 1000)
            end_of_turn = result_dict.get("end_of_turn", False)
            if end_of_turn:
                result_entry["end_of_turn"] = True

        entry["result"] = result_entry
        self._chunks.append(entry)

        if not is_listen and result_dict.get("end_of_turn", False):
            self._turn_index += 1
            self._speak_chunk_in_turn = 0

    def record_ai_audio_chunk(
        self,
        receive_ts_ms: float,
        pcm_float32: np.ndarray,
    ) -> Optional[str]:
        """记录一段 AI 产出的音频（来自 C++ _wav_poll_loop 或其他异步来源）

        以接收时间戳为锚点写入独立 timeline，供 _stitch_merged_replay 按时间对齐到右声道。
        """
        if pcm_float32 is None or len(pcm_float32) == 0:
            return None
        seq = self._ai_audio_seq
        self._ai_audio_seq += 1
        rel = f"ai_audio/turn_{self._turn_index}_stream_{seq:03d}.wav"
        path = os.path.join(self.session_dir, rel)
        self._pending_io.append(_io_pool.submit(_write_wav, path, pcm_float32, 24000))
        self._ai_audio_timeline.append({
            "seq": seq,
            "turn_index": self._turn_index,
            "receive_ts_ms": round(float(receive_ts_ms), 1),
            "rel": rel,
            "n_samples": int(len(pcm_float32)),
            "duration_ms": round(len(pcm_float32) / 24000 * 1000, 1),
        })
        return rel

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def speak_chunk_in_turn(self) -> int:
        return self._speak_chunk_in_turn

    def increment_speak_chunk(self) -> int:
        """递增当前 turn 内的 SPEAK chunk 计数器，返回当前值"""
        idx = self._speak_chunk_in_turn
        self._speak_chunk_in_turn += 1
        return idx

    def _build_recording_json(self) -> Dict[str, Any]:
        recording: Dict[str, Any] = {
            "session_id": self.session_id,
            "mode": "duplex",
            "worker_id": self.worker_id,
            "start_ts": self._start_ts.isoformat(),
            "config": self.config_snapshot,
            "chunks": self._chunks,
        }
        if self._ai_audio_timeline:
            recording["ai_audio_timeline"] = self._ai_audio_timeline
        return recording

    def _finalize_hook(self, recording: Dict[str, Any]) -> None:
        merged_rel = self._stitch_merged_replay()
        if merged_rel:
            recording["merged_replay"] = merged_rel
        video_rel = self._stitch_merged_video()
        if video_rel:
            recording["merged_replay_video"] = video_rel

    def _stitch_merged_replay(self) -> Optional[str]:
        """拼接所有 chunk 音频为立体声 WAV（left=用户 16kHz, right=AI 24kHz→16kHz）

        左声道：用户音频按 chunk back-to-back 拼接（每 chunk 至少 1.0s，模型逻辑时间）。
        右声道：优先使用 AI 独立时间轴（`_ai_audio_timeline`，duplex 标准路径），
                按 receive_ts_ms 对齐到左声道时间坐标；
                若 timeline 为空则 fallback 到 chunk.result.ai_audio（兼容旧数据）。
        """
        if not self._chunks:
            return None

        out_sr = 16000
        ai_sr = 24000

        chunk_data: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], int]] = []
        for chunk in self._chunks:
            user_pcm: Optional[np.ndarray] = None
            ai_pcm_legacy: Optional[np.ndarray] = None

            user_rel = chunk.get("user_audio")
            if user_rel:
                user_pcm = _read_wav_mono(os.path.join(self.session_dir, user_rel))

            ai_rel = (chunk.get("result") or {}).get("ai_audio")
            if ai_rel:
                raw = _read_wav_mono(os.path.join(self.session_dir, ai_rel))
                if raw is not None:
                    ai_pcm_legacy = _resample_linear(raw, ai_sr, out_sr)

            u_len = len(user_pcm) if user_pcm is not None else 0
            a_len = len(ai_pcm_legacy) if ai_pcm_legacy is not None else 0
            n = max(u_len, a_len, out_sr)
            chunk_data.append((user_pcm, ai_pcm_legacy, n))

        # 右声道独立 timeline（duplex 标准路径）
        ai_placements: List[Tuple[int, np.ndarray]] = []
        ai_end_sample = 0
        for item in self._ai_audio_timeline:
            rel = item.get("rel")
            if not rel:
                continue
            raw = _read_wav_mono(os.path.join(self.session_dir, rel))
            if raw is None or len(raw) == 0:
                continue
            ai_pcm = _resample_linear(raw, ai_sr, out_sr)
            offset = max(0, int(item.get("receive_ts_ms", 0.0) / 1000.0 * out_sr))
            ai_placements.append((offset, ai_pcm))
            ai_end_sample = max(ai_end_sample, offset + len(ai_pcm))

        user_total = sum(n for _, _, n in chunk_data)
        total = max(user_total, ai_end_sample)
        if total == 0:
            return None

        left = np.zeros(total, dtype=np.float32)
        right = np.zeros(total, dtype=np.float32)

        off = 0
        self._chunk_logical_sec: List[Tuple[float, float]] = []
        for user_pcm, ai_pcm_legacy, n in chunk_data:
            if user_pcm is not None:
                left[off:off + len(user_pcm)] = user_pcm
            # 仅在没有独立 timeline 时回退使用 chunk 上的 legacy ai_audio
            if not ai_placements and ai_pcm_legacy is not None:
                right[off:off + len(ai_pcm_legacy)] += ai_pcm_legacy
            self._chunk_logical_sec.append((off / out_sr, (off + n) / out_sr))
            off += n

        for offset, ai_pcm in ai_placements:
            end = min(offset + len(ai_pcm), total)
            if end > offset:
                right[offset:end] += ai_pcm[:end - offset]

        out_path = os.path.join(self.session_dir, "merged_replay.wav")
        _write_stereo_wav(out_path, left, right, out_sr)
        logger.info(
            f"[SessionRecorder] merged replay: {total / out_sr:.1f}s stereo WAV "
            f"(user_chunks={len(chunk_data)}, ai_stream_chunks={len(ai_placements)})"
        )
        return "merged_replay.wav"

    def _stitch_merged_video(self) -> Optional[str]:
        """Omni 模式：user_frames slideshow + merged_replay.wav + 字幕 → mp4

        使用模型逻辑时间（_chunk_logical_sec）确定帧时长和字幕时间轴，
        与 merged_replay.wav 的 back-to-back 音频完全对齐。
        """
        if self.app_type != "omni_duplex":
            return None
        if not shutil.which("ffmpeg"):
            logger.warning("[SessionRecorder] ffmpeg not found, skipping video merge")
            return None

        abs_dir = os.path.abspath(self.session_dir)
        merged_wav = os.path.join(abs_dir, "merged_replay.wav")
        if not os.path.exists(merged_wav):
            return None

        logical = getattr(self, "_chunk_logical_sec", None)

        frames: List[Tuple[float, float, str]] = []
        for i, chunk in enumerate(self._chunks):
            rel = chunk.get("user_frame")
            if not rel:
                continue
            fpath = os.path.join(abs_dir, rel)
            if not os.path.exists(fpath):
                continue
            if logical and i < len(logical):
                start_s, end_s = logical[i]
            else:
                start_s, end_s = float(len(frames)), float(len(frames) + 1)
            frames.append((start_s, end_s - start_s, fpath))
        if not frames:
            return None

        concat_path = os.path.join(abs_dir, "_frames_concat.txt")
        with open(concat_path, "w") as f:
            for i, (start_s, dur_s, fpath) in enumerate(frames):
                if i < len(frames) - 1:
                    dur_s = frames[i + 1][0] - start_s
                f.write(f"file '{fpath}'\nduration {max(dur_s, 0.04):.3f}\n")
            f.write(f"file '{frames[-1][2]}'\n")

        ass_path = self._generate_subtitles_ass()

        output = os.path.join(abs_dir, "merged_replay.mp4")
        tmp_files = [concat_path]
        if ass_path:
            tmp_files.append(ass_path)
        try:
            vf_filters = ["format=yuv420p"]
            if ass_path:
                abs_ass = os.path.abspath(ass_path)
                safe_ass = abs_ass.replace("\\", "/").replace(":", "\\:")
                vf_filters.append(f"ass='{safe_ass}'")

            subprocess.run(
                ["ffmpeg", "-y",
                 "-f", "concat", "-safe", "0", "-i", concat_path,
                 "-i", merged_wav,
                 "-vf", ",".join(vf_filters),
                 "-c:v", "libx264",
                 "-c:a", "aac", "-shortest", output],
                check=True, capture_output=True, timeout=120,
            )
            for p in tmp_files:
                try:
                    os.remove(p)
                except OSError:
                    pass
            logger.info(f"[SessionRecorder] merged video with subtitles: {output}")
            return "merged_replay.mp4"
        except Exception as e:
            logger.warning(f"[SessionRecorder] ffmpeg merge failed: {e}")
            for p in tmp_files:
                try:
                    os.remove(p)
                except OSError:
                    pass
            return None

    def _generate_subtitles_ass(self) -> Optional[str]:
        """从 chunk timeline 生成 ASS 字幕文件（模型逻辑时间）

        字幕逻辑：
        - 同 turn 内 SPEAK chunk 的 text 累积显示
        - 起止时间由 _chunk_logical_sec 决定（与音频/视频对齐）
        - turn 结束后字幕再保持 1.5s 然后消失

        Returns:
            ASS 文件路径，无字幕内容时返回 None
        """
        logical = getattr(self, "_chunk_logical_sec", None)
        events: List[Dict[str, Any]] = []
        accum_text = ""
        in_turn = False

        for i, chunk in enumerate(self._chunks):
            result = chunk.get("result", {})
            mode = result.get("mode", "LISTEN")
            text = result.get("text", "")
            end_of_turn = result.get("end_of_turn", False)

            if logical and i < len(logical):
                chunk_start = logical[i][0]
                next_start = logical[i + 1][0] if i + 1 < len(logical) else logical[i][1]
            else:
                chunk_start = float(i)
                next_start = float(i + 1)

            if mode == "SPEAK" and text:
                if not in_turn:
                    accum_text = text
                    in_turn = True
                else:
                    accum_text += text

                start_s = chunk_start
                if end_of_turn:
                    end_s = next_start + 1.5
                    in_turn = False
                else:
                    end_s = next_start

                events.append({
                    "start": start_s,
                    "end": end_s,
                    "text": accum_text,
                })

        if not events:
            return None

        merged_events = self._merge_subtitle_events(events)

        ass_path = os.path.join(self.session_dir, "_subtitles.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("[Script Info]\n")
            f.write("ScriptType: v4.00+\n")
            f.write("PlayResX: 1280\n")
            f.write("PlayResY: 720\n")
            f.write("WrapStyle: 0\n")
            f.write("\n[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
                    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
                    "MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: Default,Noto Sans CJK SC,42,&H00FFFFFF,&H000000FF,&H00000000,"
                    "&HA0000000,-1,0,0,0,100,100,0,0,3,1,2,2,30,30,40,1\n")
            f.write("\n[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
                    "MarginV, Effect, Text\n")
            for ev in merged_events:
                start_str = self._ass_timestamp(ev["start"])
                end_str = self._ass_timestamp(ev["end"])
                safe_text = ev["text"].replace("\n", "\\N")
                f.write(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{safe_text}\n")

        logger.info(f"[SessionRecorder] generated {len(merged_events)} subtitle events")
        return ass_path

    @staticmethod
    def _merge_subtitle_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并时间相邻的字幕事件（同一 turn 内文本累积，后者包含前者的全部文本）

        连续的累积事件会合并为：前一个事件的 end = 下一个事件的 start，
        最后一个事件保持到其 end 时间。这样字幕平滑切换，没有闪烁。
        """
        if not events:
            return []
        merged: List[Dict[str, Any]] = []
        for i, ev in enumerate(events):
            if i + 1 < len(events) and events[i + 1]["text"].startswith(ev["text"]):
                merged.append({
                    "start": ev["start"],
                    "end": events[i + 1]["start"],
                    "text": ev["text"],
                })
            else:
                merged.append(ev)
        return merged

    @staticmethod
    def _ass_timestamp(seconds: float) -> str:
        """秒数转 ASS 时间格式 H:MM:SS.cc"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds - int(seconds)) * 100))
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


class TurnBasedSessionRecorder(SessionRecorder):
    """Turn-based 模式录制器（Streaming / Chat）

    逐 turn 记录 timeline。一个 turn = 一次 prefill + generate 循环。
    Streaming 模式中 generate 产出多个 chunk，累积后合并为一条 turn 记录。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._turns: List[Dict[str, Any]] = []
        self._current_turn: Optional[Dict[str, Any]] = None
        self._current_text_parts: List[str] = []
        self._current_audio_parts: List[np.ndarray] = []
        self._image_counter: int = 0

    def start_turn(
        self,
        turn_index: int,
        request_ts_ms: float,
        input_summary: Dict[str, Any],
    ) -> None:
        """开始新 turn

        Args:
            turn_index: turn 序号
            request_ts_ms: 相对于 session 开始的请求时间 (ms)
            input_summary: 输入摘要 (messages 的序列化表示，不含原始二进制)
        """
        self._current_turn = {
            "turn_index": turn_index,
            "request_ts_ms": round(request_ts_ms, 1),
            "input": input_summary,
        }
        self._current_text_parts = []
        self._current_audio_parts = []

    def add_streaming_chunk(self, text_delta: Optional[str], audio_base64: Optional[str]) -> None:
        """累积一个 streaming chunk 的文本和音频

        Args:
            text_delta: 文本增量（可选）
            audio_base64: base64 编码的 float32 PCM 24kHz（可选）
        """
        if text_delta:
            self._current_text_parts.append(text_delta)
        if audio_base64:
            try:
                audio_bytes = base64.b64decode(audio_base64)
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                self._current_audio_parts.append(audio_np)
            except Exception as e:
                logger.warning(f"[TurnBasedRecorder] failed to decode audio chunk: {e}")

    def end_turn(self, timing: Dict[str, Any]) -> None:
        """结束当前 turn，保存累积的音频，添加到 timeline

        Args:
            timing: 性能计时数据 (elapsed_ms, token_stats, ...)
        """
        if self._current_turn is None:
            logger.warning("[TurnBasedRecorder] end_turn called without start_turn")
            return

        turn_index = self._current_turn["turn_index"]
        full_text = "".join(self._current_text_parts)

        output: Dict[str, Any] = {"text": full_text}

        if self._current_audio_parts:
            combined = np.concatenate(self._current_audio_parts)
            audio_rel = self.save_ai_audio_turn(turn_index, combined)
            output["audio"] = audio_rel
            output["audio_duration_ms"] = round(len(combined) / 24000 * 1000)

        self._current_turn["output"] = output
        self._current_turn["timing"] = timing
        self._turns.append(self._current_turn)

        self._current_turn = None
        self._current_text_parts = []
        self._current_audio_parts = []

    def record_chat_turn(
        self,
        turn_index: int,
        request_ts_ms: float,
        input_summary: Dict[str, Any],
        output_text: str,
        output_audio: Optional[np.ndarray],
        timing: Dict[str, Any],
    ) -> None:
        """一步完成 Chat 模式的 turn 记录（无 streaming chunk 累积）

        Args:
            turn_index: turn 序号
            request_ts_ms: 请求时间 (ms)
            input_summary: 输入摘要
            output_text: 输出文本
            output_audio: 输出音频 (float32 PCM, 可选)
            timing: 性能计时
        """
        output: Dict[str, Any] = {"text": output_text}
        if output_audio is not None and len(output_audio) > 0:
            audio_rel = self.save_ai_audio_turn(turn_index, output_audio)
            output["audio"] = audio_rel
            output["audio_duration_ms"] = round(len(output_audio) / 24000 * 1000)

        self._turns.append({
            "turn_index": turn_index,
            "request_ts_ms": round(request_ts_ms, 1),
            "input": input_summary,
            "output": output,
            "timing": timing,
        })

    def next_image_index(self) -> int:
        """获取下一个可用的图片索引"""
        idx = self._image_counter
        self._image_counter += 1
        return idx

    def _build_recording_json(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.app_type,
            "worker_id": self.worker_id,
            "start_ts": self._start_ts.isoformat(),
            "config": self.config_snapshot,
            "turns": self._turns,
        }
