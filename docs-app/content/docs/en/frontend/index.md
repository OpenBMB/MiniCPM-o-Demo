---
title: "Frontend Module Overview"
---

The frontend is built with pure **HTML + JavaScript + CSS**, with no framework dependencies, and uses a modular design to organize code. It supports real-time WebSocket communication, Web Audio API audio processing, and MediaRecorder video recording.

## Module Structure

```
static/
├── index.html                          # Home page (mode selection + recent sessions)
├── turnbased.html                      # Turn-based chat page
├── admin.html                          # Admin panel
├── session-viewer.html                 # Session replay viewer
│
├── omni/                               # Omni full-duplex page
│   ├── omni.html / omni-app.js / omni.css
│
├── audio-duplex/                       # Audio full-duplex page
│   ├── audio_duplex.html / audio-duplex-app.js / audio-duplex.css
│
├── duplex/                             # Duplex shared library
│   ├── duplex-shared.css               #   Shared styles
│   ├── lib/                            #   Core library (10+ modules)
│   └── ui/                             #   UI components
│
├── shared/                             # Cross-page shared components
│   ├── app-nav.js / preset-selector.js / save-share.js
│
├── lib/                                # Common utility library
│   ├── chat-eta-estimator.js / countdown-timer.js
│
├── ref-audio-player.js                 # Reference audio player
├── system-content-editor.js            # System content editor
└── user-content-editor.js              # User message editor
```

## Sub-document Navigation

| Document | Content |
|----------|---------|
| [Pages & Routing](./pages/) | Detailed page functionality, routing system, Turn-based Chat state management |
| [Audio Processing](./audio/) | AudioWorklet capture, AudioPlayer playback, LUFS measurement, mixer |
| [Duplex Session](./duplex-session/) | DuplexSession class, WebSocket protocol, state machine, recording system |
| [UI Components](./components/) | Shared component library, content editors, preset selector, navigation system |

## Page Routes

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Mode selection cards, recent session list |
| Turn-based Chat | `/turnbased` | Turn-based Chat interaction |
| Omni Full-Duplex | `/omni` | Vision + voice full-duplex |
| Audio Full-Duplex | `/audio_duplex` | Audio-only full-duplex |
| Admin Panel | `/admin` | Worker status, queue management, app toggles |
| Session Replay | `/s/{session_id}` | Session recording playback |
| API Docs | `/docs` | FastAPI auto-generated |

## Tech Stack

- No framework: pure HTML + ES Module JavaScript
- Real-time communication: WebSocket (full-duplex, streaming chat)
- Audio processing: Web Audio API + AudioWorklet
- Video capture: getUserMedia + Canvas
- Video recording: MediaRecorder API
- State persistence: localStorage
