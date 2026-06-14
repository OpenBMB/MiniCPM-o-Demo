---
title: "Audio Full-Duplex Protocol"
description: "Current audio full-duplex Realtime API protocol"
---

The current audio full-duplex protocol is documented here:

[Audio Full-Duplex](./realtime-api/audio/)

Use:

```text
wss://host/v1/realtime?mode=audio
```

The current protocol uses `session.init`, `input.append`, and `response.output.delta`.
