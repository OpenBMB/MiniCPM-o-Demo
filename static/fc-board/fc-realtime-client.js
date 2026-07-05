export class FcRealtimeClient {
  constructor({ onEvent, onStatus, onSend } = {}) {
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.onSend = onSend;
    this.ws = null;
    this.ready = false;
    this.queued = false;
    this._pendingInit = null;
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
    if (this.queued) {
      this._send({ type: 'session.init', payload });
    } else {
      this._pendingInit = payload;
    }
  }

  appendAudio({ audioBase64, sampleRate = 16000 }) {
    this._send({
      type: 'input.append',
      input: {
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

  _send(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.onSend?.(message);
    this.ws.send(JSON.stringify(message));
  }

  _handleMessage(message) {
    const event = JSON.parse(message.data);
    if (event.type === 'session.queue_done') {
      this.queued = true;
      if (this._pendingInit) {
        const payload = this._pendingInit;
        this._pendingInit = null;
        this._send({ type: 'session.init', payload });
      }
    }
    if (event.type === 'session.created') {
      this.ready = true;
    }
    this.onEvent?.(event);
  }
}
