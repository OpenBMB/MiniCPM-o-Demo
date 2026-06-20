---
title: "Realtime API Protocol"
description: "Current MiniCPM-o Realtime API protocol entry"
---

The Realtime API protocol is documented under the Realtime API section.

- [Overview](./realtime-api/overview/)
- [Chat mode](./realtime-api/chat/)
- [Video full-duplex](./realtime-api/video/)
- [Audio full-duplex](./realtime-api/audio/)
- [Examples](./realtime-api/examples/)

The current public WebSocket endpoint is:

```text
wss://host/v1/realtime?mode={chat|video|audio}
```

Public clients connect to the Gateway. The Gateway forwards sessions through a Python Worker to the
actual backend. `mode=chat` maps to backend runtime mode `turn_based`; `mode=video` and `mode=audio`
map to `full_duplex`.

The current protocol uses `session.init`, `input.append`, and `response.output.delta`.
