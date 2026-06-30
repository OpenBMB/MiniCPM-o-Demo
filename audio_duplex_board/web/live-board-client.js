export class LiveBoardClient {
  constructor({ onEvent, onStatus }) {
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.ws = null;
  }

  async connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws/session`;
    this.ws = new WebSocket(url);
    this.ws.onmessage = (message) => {
      this.onEvent(JSON.parse(message.data));
    };
    this.ws.onclose = () => this.onStatus?.('WebSocket closed');
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = () => reject(new Error('WebSocket connection failed'));
    });
  }

  send(type, payload = {}) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.ws.send(JSON.stringify({ type, payload }));
  }

  close() {
    if (this.ws) this.ws.close();
  }
}
