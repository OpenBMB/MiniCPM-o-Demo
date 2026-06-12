---
title: "Video Full-Duplex Protocol"
description: "Current video full-duplex Realtime API protocol"
---

The current video full-duplex protocol is documented here:

[Video Full-Duplex](./realtime-api/video/)

Use:

```text
wss://host/v1/realtime?mode=video
```

The current protocol uses `session.init`, `input.append`, and `response.output.delta`.
