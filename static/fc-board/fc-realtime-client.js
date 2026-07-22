export class FcRealtimeClient {
  constructor({ onEvent, onStatus, onSend } = {}) {
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.onSend = onSend;
    this.ws = null;
    this.ready = false;
    this.queued = false;
    this._pendingSessionFrame = null;
    this.history = [];
    this.resumeIdentity = null;
    this._inputSeq = 0;
  }

  async connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/v1/realtime?mode=audio`;
    this.ws = new WebSocket(url);
    this.ws.onmessage = (message) => this._handleMessage(message);
    this.ws.onclose = () => {
      this.ready = false;
      this.onStatus?.('WebSocket closed');
    };
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error('WebSocket connection failed'));
    });
  }

  initSession(payload) {
    const frame = { type: 'session.init', payload };
    if (this.queued) {
      this._send(frame);
    } else {
      this._pendingSessionFrame = frame;
    }
  }

  resumeSession(payload) {
    const priorHistory = Array.isArray(payload?.history) ? payload.history : [];
    this.history = priorHistory.map((frame) => ({
      dir: (
        frame?.type === 'session.init'
        || frame?.type === 'input.append'
        || String(frame?.type || '').startsWith('input.')
      ) ? 'up' : 'down',
      frame: JSON.parse(JSON.stringify(frame)),
    }));
    this.resumeIdentity = {
      protocol_version: payload.protocol_version,
      model: payload.model,
      tokenizer_target: payload.tokenizer_target,
      tokenizer_fingerprint: payload.tokenizer_fingerprint,
      ref_audio_sha256: payload.ref_audio_sha256,
      prompt_wav_sha256: payload.prompt_wav_sha256,
    };
    this._inputSeq = priorHistory.filter(
      (frame) => frame?.type === 'input.append',
    ).length;
    const frame = { type: 'session.resume', payload };
    if (this.queued) {
      this._send(frame);
    } else {
      this._pendingSessionFrame = frame;
    }
  }

  buildResumePayload(throughUnitIndex) {
    const checkpointIndex = this.history.findIndex((item) => (
      item.dir === 'down'
      && item.frame.type === 'response.unit.committed'
      && item.frame.unit_index === throughUnitIndex
    ));
    if (checkpointIndex < 0) {
      throw new Error(`Missing Unit checkpoint: ${throughUnitIndex}`);
    }
    const checkpoint = this.history[checkpointIndex].frame;
    if (checkpoint.resume?.status !== 'available') {
      throw new Error(
        `Unit ${throughUnitIndex} is not resumable: ${checkpoint.resume?.reason || 'unknown'}`,
      );
    }
    if (!this.resumeIdentity) {
      throw new Error('Missing session resume identity');
    }
    return {
      ...this.resumeIdentity,
      through_unit_index: throughUnitIndex,
      history: this.history
        .slice(0, checkpointIndex + 1)
        .map((item) => item.frame),
    };
  }

  appendAudio({ audioBase64, sampleRate = 16000, inputId = null }) {
    const resolvedInputId = inputId || `input_${String(this._inputSeq).padStart(6, '0')}`;
    this._inputSeq += 1;
    this._send({
      type: 'input.append',
      input: {
        input_id: resolvedInputId,
        audio_base64: audioBase64,
        sample_rate: sampleRate,
      },
    });
  }

  sendToolResult({ toolCallId, contents }) {
    this._send({
      type: 'input.tool_result',
      tool_call_id: toolCallId,
      contents,
    });
  }

  close(reason = 'client_closed') {
    if (!this.ws) return;
    if (this.ws.readyState === WebSocket.OPEN) {
      this._send({ type: 'session.close', reason });
    } else {
      this.ws.close();
    }
  }

  disconnect(reason = 'network_reconnect') {
    if (!this.ws) return;
    this.ws.close(1000, reason);
  }

  _send(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.history.push({
      dir: 'up',
      frame: JSON.parse(JSON.stringify(message)),
    });
    this.onSend?.(message);
    this.ws.send(JSON.stringify(message));
  }

  _handleMessage(message) {
    const event = JSON.parse(message.data);
    if (
      event.type === 'session.created'
      || event.type === 'response.generation.step_batch'
      || event.type === 'response.unit.committed'
      || event.type.startsWith('response.think.')
      || event.type.startsWith('response.tool_call.')
      || event.type === 'response.output.delta'
      || event.type === 'response.output.sp_tokens'
      || event.type === 'response.warning'
    ) {
      this.history.push({
        dir: 'down',
        frame: JSON.parse(JSON.stringify(event)),
      });
    }
    if (event.type === 'session.queue_done') {
      this.queued = true;
      if (this._pendingSessionFrame) {
        const frame = this._pendingSessionFrame;
        this._pendingSessionFrame = null;
        this._send(frame);
      }
    }
    if (event.type === 'session.created' || event.type === 'session.resumed') {
      this.ready = true;
    }
    if (event.type === 'session.created' && event.resume) {
      this.resumeIdentity = { ...event.resume };
    }
    this.onEvent?.(event);
  }
}
