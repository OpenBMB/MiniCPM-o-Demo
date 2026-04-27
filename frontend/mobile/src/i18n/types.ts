export type Lang = 'zh' | 'en'

export interface Translations {
  // ---------- general ----------
  newChat: string
  settings: string
  cancel: string
  close: string
  confirm: string
  delete: string
  share: string
  copy: string
  copied: string
  upload: string
  play: string
  pause: string
  stop: string
  default_: string
  clear: string
  record: string
  stopRecording: string
  retry: string
  save: string
  loading: string

  // ---------- top bar ----------
  openMenu: string
  enterAudioDuplex: string
  enterVideoDuplex: string

  // ---------- session ----------
  sessionTitle_voice: string
  sessionTitle_image: string
  sessionTitle_audio: string
  sessionTitle_video: string
  justNow: string
  minutesAgo: (n: number) => string
  yesterday: string
  noHistoryYet: string
  createNewChat: string
  deleteSessionConfirm: (title: string) => string
  clearAllDataConfirm: string
  clearAllData: string
  historySessions: string

  // ---------- service status ----------
  checkingBackend: string
  backendReady: string
  gatewayDegraded: string
  backendUnreachable: string
  queueHint: (pos: number, eta: number) => string

  // ---------- composer ----------
  placeholder: string
  holdToTalk: string
  releaseToSend: string
  releaseToCancel: string
  speaking: string
  switchToKeyboard: string
  switchToVoice: string
  openAttachMenu: string
  closeAttachMenu: string
  sendMessage: string
  stopGeneration: string
  selectAttachment: string
  removeAttachment: string
  takePhoto: string
  camera: string
  album: string
  files: string
  phoneCall: string

  // ---------- message actions ----------
  stopPlayback: string
  readAloud: string
  regenerate: string
  interrupted: string

  // ---------- recording ----------
  micNotReady: string
  micInitFailed: string
  audioChannelFailed: string
  recordingFailed: (err: string) => string

  // ---------- attachment errors ----------
  fileTooLarge: (kind: string, size: string, max: string) => string
  videoTooLong: (duration: number, max: number) => string
  attachProcessFailed: (err: string) => string
  imageLabel: string
  audioLabel: string
  videoLabel: string

  // ---------- network / generation ----------
  thinking: string
  generating: string
  emptyReply: string
  requestFailed: string
  requestFailedDetail: (err: string) => string
  stoppedReply: string
  sendFailed: (err: string) => string
  connectFailed: (err: string) => string
  wsError: string
  wsClosed: string

  // ---------- settings sheet ----------
  currentParams: string
  refAudio: string
  refAudioNotSet: string
  refAudioSource: string
  presetRefAudio: string
  defaultRefAudio: string
  systemPrompt: string
  params: string
  lengthPenalty: string
  maxTokens: string
  voiceReply: string
  streamingOutput: string
  turnBased: string
  audioDuplex: string
  videoDuplex: string
  preset: string
  saveCurrentPreset: string
  noPresetsYet: string
  deletePresetConfirm: (name: string) => string
  notSet: string
  on: string
  off: string

  // ---------- ref audio ----------
  refAudioDefault: string
  refAudioRecordTooShort: string
  refAudioRecordFailed: string
  refAudioRecordDuration: (s: string) => string
  refAudioRecorded: (s: string) => string
  refAudioMicUnsupported: string
  refAudioRecordUnsupported: string
  refAudioRecordError: (err: string) => string
  refAudioNoDefault: string
  refAudioNoPlayable: string
  refAudioProcessFailed: (err: string) => string

  // ---------- share ----------
  shareChat: string
  noBackendRecord: string
  shareTitle: string
  shareHint: string
  linkLabel: string
  sessionLabel: string
  commentPlaceholder: string
  sharing: string
  copiedToClipboard: (url: string) => string
  copyManually: (url: string) => string
  shareFailed: (err: string) => string

  // ---------- preset ----------
  custom: string
  presetSaved: (name: string) => string
  presetNamePrompt: string

  // ---------- duplex ----------
  audioCall: string
  callSubtitles: string
  closeSubtitles: string
  subtitlesWillAppear: string
  forceListen: string
  forceListening: string
  paused: string
  inCall: string
  preparing: string
  queuing: string
  connecting: string
  error: string
  notConnected: string
  speakToStart: string
  tapToStart: string
  listening: string
  continue_: string
  endCall: string
  startCall: string
  shareCall: string
  shareCallHint: string

  // ---------- duplex (video) ----------
  live: string
  stopped: string
  start: string
  mirrorFlip: string
  flipCamera: string
  subtitlesOnOff: string
  hideSubtitles: string
  showSubtitles: string
  exit: string
  hd: string

  // ---------- duplex state ----------
  duplexInProgress: (mode: string) => string
  duplexEnded: (mode: string) => string
  duplexWaiting: string
  tapStartDuplex: string
  requestingMicCamera: string
  requestingMic: string
  workerAssigned: (mode: string) => string
  sessionReady: (mode: string) => string
  duplexPausing: (mode: string) => string
  duplexPaused: (mode: string) => string
  startFailed: (err: string) => string
  flipCameraFailed: (err: string) => string
  cameraPreviewFailed: (err: string) => string

  // ---------- browser compat ----------
  browserNoGetUserMedia: string
  browserNoAudioContext: string
  browserNoCamera: string
}
