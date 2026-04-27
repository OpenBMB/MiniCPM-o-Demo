/**
 * Lightweight i18n module for the desktop pages.
 *
 * Language detection priority:
 *   1. URL search param  ?lang=zh  /  ?lang=en
 *   2. localStorage('minicpmo_lang')
 *   3. navigator.language
 *   4. fallback → 'zh'
 *
 * Usage:
 *   import { lang, t, setLang } from '/static/shared/i18n.js';
 *   someElement.textContent = t.send;
 *   setLang('en');               // hot-switch + page refresh
 *   setLang('en', false);        // hot-switch, no reload
 */

const STORAGE_KEY = 'minicpmo_lang';

// ─── Translations ──────────────────────────────────────────────────────

const zh = {
  // nav / home
  home: 'Home',
  chooseMode: '选择模式',
  recentSessions: '最近会话（仅你可见）',
  clearRecent: '清除',
  enter: '进入 →',
  mobile: 'Mobile',
  mobileName: '移动端',
  mobileDesc: 'React + Vite 移动端交互界面，支持单工和全双工。',
  turnbasedDesc: '多轮对话：文本和音频输入，流式输出。适合问答、编辑和多轮任务。',
  omniDesc: '全双工实时：摄像头 + 麦克风或本地视频/音频文件；实时对话。',
  audioDuplexDesc: '纯语音全双工：麦克风或本地音频文件，低延迟语音对话。',

  // header / status
  connecting: '连接中…',
  offline: '离线',

  // turnbased page
  turnbasedChat: '单工对话',
  omniFullDuplex: '视频全双工',
  advancedSystemSettings: '高级系统设置',
  systemContent: '系统内容（文本+音频，可排序）',
  ttsRefAudio: 'TTS 参考音频（声音克隆）',
  extractFromSystem: '从系统提取',
  independent: '独立上传',
  holdSpaceToRecord: '按住 Space 录音…',
  enterToSend: 'Enter 发送',
  dragDropHint: '拖放或 ⌘V 粘贴附件',
  dropOverlayText: '拖放图片、音频或视频到此处',
  streaming: '流式输出',
  voiceResponse: '语音回复',
  newTokens: '新 token 数',
  lenPen: '长度惩罚',
  recordAudio: '录音（Space 键）',
  systemSettings: '系统设置',
  exportMsgpack: '导出 .msgpack',
  generating: '生成中…',
  cancelGeneration: '✕ 取消',
  typeOrRecord: '输入或按 Space 录音…',
  noMessagesToExport: '没有消息可导出。',

  // omni / audio-duplex common
  resetSettings: '重置设置',
  resetConfirm: '确定恢复所有设置为默认值？',
  cameraLoading: '摄像头加载中…',
  flipCamera: '翻转摄像头',
  mirrorFlip: '镜像翻转',
  fullscreen: '全屏',
  subtitlesOnOff: '字幕开关',
  live: '在线',
  forceListen: '强制收听',
  hd: '高清',
  start: '开始',
  pause_: '暂停',
  stop_: '停止',
  liveMode: '实时',
  fileMode: '文件',
  chooseVideo: '选择视频',
  chooseAudio: '选择音频',
  noFile: '暂无文件',
  audio: '音频',
  source: '源',
  videoOnly: '仅视频',
  micOnly: '仅麦克风',
  videoMic: '视频+麦克风',
  playbackVolume: '播放音量',
  mixer: '混音器',
  video: '视频',
  sampleT: '采样 t',
  padS: '补空 s',

  // response length
  responseLengthControls: '回复长度控制',
  lengthPenalty: '长度惩罚',
  verbose: '详细',
  balanced: '平衡',
  concise: '简洁',
  verboseHint: '越长越详细…',
  balancedHint: '适中…',
  conciseHint: '短小精悍…',

  // vision
  visionSettings: '视觉设置',
  hdVision: '高清视觉',

  // other settings
  otherSettings: '其他设置',
  delay: '延迟',
  ms: 'ms',
  maxKV: '最大 KV',
  tok: 'tok',
  maxKVHint: '≤ 8192（最大训练长度）',

  // audio devices
  audioDevices: '音频设备',
  mic: '麦克风',
  speaker: '扬声器',
  refresh: '刷新',

  // subtitle
  fullscreenSubtitle: '全屏字幕',
  height: '高度',
  percentFromBottom: '距底 %',
  opacityPercent: '不透明度 %',

  // system prompt
  systemPromptDetails: '系统提示词详情',
  systemPromptPlaceholder: '系统提示词…',
  llmRefAudio: 'LLM 参考音频（系统提示词）',
  sameAsLLM: '与 LLM 相同',

  // conversation
  conversation: '对话',
  startSessionToSee: '开始会话后即可查看对话',
  startVoiceConversation: '开始会话以进行实时语音对话',

  // controls
  rec: '录制',
  downloadRec: '下载录制',

  // share (desktop)
  shareDialog: '分享对话',
  shareHint: '将录制上传到服务器并复制分享链接。',
  commentOptional: '评语（可选）',
  cancel: '取消',
  share: '分享',
  uploadAndShare: '上传并分享',
  uploading: '上传中…',
  copiedToClipboard: '已复制到剪贴板',
  shareLink: '分享链接：',
  commentSaveFailed: '评语保存失败：',
  uploadFailed: '上传失败：',

  // ref-audio-player
  noRefAudio: '暂无参考音频',
  upload: '上传',
  resetToDefault: '恢复默认',
  audioTooLong: (dur, max) => `音频过长：${dur}（最大 ${max}s）`,
  processAudioFailed: (err) => `处理音频失败：${err}`,

  // queue
  workers: '工作节点',
  idle: '空闲',
  queue: '队列',
  queued: (pos, eta) => `排队 ${pos}，约 ${eta}s`,
  queuedOvertime: (pos, overtime) => `排队 ${pos}，超时 +${overtime}s`,

  // metrics
  total: '总计',
  stopped: '[已停止]',

  // misc
  edit: '编辑',
  resendFromHere: '从此处重发',
  clickToCopy: '点击复制：',
  copied: '已复制！',

  // audio duplex
  audioDuplexTitle: '音频双工',
  experimentalNotice: '实验性功能：推荐使用耳机以避免回声。',
  dismiss: '关闭',
  listeningHint: '正在收听…',
  micWaveform: '麦克风波形',
  fileOnly: '仅文件',
  fileMic: '文件+麦克风',
  padding: '补空',
  session: '会话',
  selectAudioFirst: '请先选择一个音频文件。',
  selectVideoFirst: '请先选择一个视频文件。',

  // mobile-omni
  backToMobile: '返回移动端',
  settings: '设置',
  openSettings: '打开设置',
  shareCallLabel: '分享通话',
  shareCallAvailableAfter: '通话开始后可分享',
  shareCallUploadHint: '上传录制并复制链接，可附评语',
  toggleFlashlight: '切换闪光灯',

  // preset selector
  presetTitle: '预设系统提示词',
  presetSubtitle: '控制回复语言、音色风格、节奏和音质。可通过"高级"进一步编辑系统提示词。Web 界面不会替换 C++ TTS 缓存（\`prompt_cache.gguf\`）；如需完整替换 TTS 音色，请参考<a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/%E6%8D%A2%E9%9F%B3%E8%89%B2%E6%8C%87%E5%8D%97.md" target="_blank" rel="noopener">换音色指南</a>。',
  presetAdvanced: '高级',
  presetAdvancedTooltip: '显示/隐藏系统提示词详情以便自定义',
  presetLoadingMedia: '正在加载预设素材…',

  // language
  language: '语言',
  langZh: '中文',
  langEn: 'English',
};

const en = {
  // nav / home
  home: 'Home',
  chooseMode: 'Choose Mode',
  recentSessions: 'Recent Sessions (Only visible to you)',
  clearRecent: 'Clear',
  enter: 'Enter →',
  mobile: 'Mobile',
  mobileName: 'Mobile',
  mobileDesc: 'React + Vite mobile interface for turn-based and full-duplex modes.',
  turnbasedDesc: 'Multi-turn chat: text and audio input with streaming output. For Q&A, editing, and multi-turn tasks.',
  omniDesc: 'Full-duplex real-time: camera + mic or local video/audio file; speak and see with live replies.',
  audioDuplexDesc: 'Voice-only full-duplex: live mic or local audio file, low-latency voice conversation.',

  // header / status
  connecting: 'Connecting…',
  offline: 'Offline',

  // turnbased page
  turnbasedChat: 'Turn-based Chat',
  omniFullDuplex: 'Omni Full-Duplex',
  advancedSystemSettings: 'Advanced System Settings',
  systemContent: 'System Content (text + audio, reorderable)',
  ttsRefAudio: 'TTS Ref Audio (voice cloning)',
  extractFromSystem: 'Extract from System',
  independent: 'Independent',
  holdSpaceToRecord: 'Hold Space to record…',
  enterToSend: 'Enter to send',
  dragDropHint: 'Drag & drop or ⌘V to attach',
  dropOverlayText: 'Drop image, audio or video here',
  streaming: 'Streaming',
  voiceResponse: 'Voice Response',
  newTokens: 'new tokens',
  lenPen: 'len_pen',
  recordAudio: 'Record audio (Space key)',
  systemSettings: 'System Settings',
  exportMsgpack: 'Export .msgpack',
  generating: 'Generating…',
  cancelGeneration: '✕ Cancel',
  typeOrRecord: 'Type or press Space to record…',
  noMessagesToExport: 'No messages to export.',

  // omni / audio-duplex common
  resetSettings: 'Reset Settings',
  resetConfirm: 'Reset all settings to defaults?',
  cameraLoading: 'Camera loading…',
  flipCamera: 'Flip camera',
  mirrorFlip: 'Mirror flip',
  fullscreen: 'Fullscreen',
  subtitlesOnOff: 'Subtitles on/off',
  live: 'LIVE',
  forceListen: 'Force Listen',
  hd: 'HD',
  start: 'Start',
  pause_: 'Pause',
  stop_: 'Stop',
  liveMode: 'Live',
  fileMode: 'File',
  chooseVideo: 'Choose video',
  chooseAudio: 'Choose audio',
  noFile: 'No file',
  audio: 'Audio',
  source: 'Source',
  videoOnly: 'Video-only',
  micOnly: 'Mic-only',
  videoMic: 'Video+Mic',
  playbackVolume: 'Playback Volume',
  mixer: 'Mixer',
  video: 'Video',
  sampleT: 'Sample t',
  padS: 'Pad s',

  // response length
  responseLengthControls: 'Response Length Controls',
  lengthPenalty: 'Length Penalty',
  verbose: 'Verbose',
  balanced: 'Balanced',
  concise: 'Concise',
  verboseHint: 'Longer, more detailed…',
  balancedHint: 'Moderate…',
  conciseHint: 'Short, concise…',

  // vision
  visionSettings: 'Vision Settings',
  hdVision: 'High-Definition Vision',

  // other settings
  otherSettings: 'Other Settings',
  delay: 'Delay',
  ms: 'ms',
  maxKV: 'MaxKV',
  tok: 'tok',
  maxKVHint: '≤ 8192 (max training length)',

  // audio devices
  audioDevices: 'Audio Devices',
  mic: 'Mic',
  speaker: 'Speaker',
  refresh: 'Refresh',

  // subtitle
  fullscreenSubtitle: 'Fullscreen Subtitle',
  height: 'Height',
  percentFromBottom: '% from bottom',
  opacityPercent: 'Opacity %',

  // system prompt
  systemPromptDetails: 'System Prompt Details',
  systemPromptPlaceholder: 'System prompt…',
  llmRefAudio: 'LLM Ref Audio (system prompt)',
  sameAsLLM: 'Same as LLM',

  // conversation
  conversation: 'Conversation',
  startSessionToSee: 'Start a session to see the conversation',
  startVoiceConversation: 'Start a session to begin real-time voice conversation',

  // controls
  rec: 'Rec',
  downloadRec: 'Download Rec',

  // share (desktop)
  shareDialog: 'Share Conversation',
  shareHint: 'Upload session recording to server and copy shareable link.',
  commentOptional: 'Comment (optional)',
  cancel: 'Cancel',
  share: 'Share',
  uploadAndShare: 'Upload & Share',
  uploading: 'Uploading…',
  copiedToClipboard: 'Copied to clipboard',
  shareLink: 'Share link: ',
  commentSaveFailed: 'Comment save failed: ',
  uploadFailed: 'Upload failed: ',

  // ref-audio-player
  noRefAudio: 'No reference audio',
  upload: 'Upload',
  resetToDefault: 'Reset to default',
  audioTooLong: (dur, max) => `Audio too long: ${dur} (max ${max}s)`,
  processAudioFailed: (err) => `Failed to process audio: ${err}`,

  // queue
  workers: 'Workers',
  idle: 'idle',
  queue: 'Queue',
  queued: (pos, eta) => `Queued ${pos}, ~${eta}s remaining`,
  queuedOvertime: (pos, overtime) => `Queued ${pos}, worker busy, overtime +${overtime}s`,

  // metrics
  total: 'Total',
  stopped: '[stopped]',

  // misc
  edit: 'Edit',
  resendFromHere: 'Resend from here',
  clickToCopy: 'Click to copy: ',
  copied: 'Copied!',

  // audio duplex
  audioDuplexTitle: 'Audio Full-Duplex',
  experimentalNotice: 'Experimental feature: headphones recommended to avoid echo.',
  dismiss: 'Dismiss',
  listeningHint: 'Listening…',
  micWaveform: 'Microphone waveform',
  fileOnly: 'File-only',
  fileMic: 'File+Mic',
  padding: 'Padding',
  session: 'Session',
  selectAudioFirst: 'Please select an audio file first.',
  selectVideoFirst: 'Please select a video file first.',

  // mobile-omni
  backToMobile: 'Back to mobile',
  settings: 'Settings',
  openSettings: 'Open settings',
  shareCallLabel: 'Share call',
  shareCallAvailableAfter: 'Available after call starts',
  shareCallUploadHint: 'Upload recording and copy link, add optional comment',
  toggleFlashlight: 'Toggle flashlight',

  // preset selector
  presetTitle: 'Preset System Prompt',
  presetSubtitle: 'Controls response language, voice style, rhythm and timbre. You can further edit the system prompt text in Advanced. The web UI does not replace the C++ TTS cache (\`prompt_cache.gguf\`); for full TTS voice replacement, see the <a href="https://github.com/OpenSQZ/MiniCPM-V-CookBook/blob/main/deployment/llama.cpp-omni/%E6%8D%A2%E9%9F%B3%E8%89%B2%E6%8C%87%E5%8D%97.md" target="_blank" rel="noopener">voice guide</a>.',
  presetAdvanced: 'Advanced',
  presetAdvancedTooltip: 'Show/hide system prompt details for customization',
  presetLoadingMedia: 'Loading preset media…',

  // language
  language: 'Language',
  langZh: '中文',
  langEn: 'English',
};

// ─── Runtime ───────────────────────────────────────────────────────────

const bundles = { zh, en };

function _detect() {
  try {
    const p = new URLSearchParams(location.search);
    const u = p.get('lang');
    if (u === 'en' || u === 'zh') return u;
  } catch { /* */ }
  try {
    const s = localStorage.getItem(STORAGE_KEY);
    if (s === 'en' || s === 'zh') return s;
  } catch { /* */ }
  try {
    if (navigator.language.toLowerCase().startsWith('en')) return 'en';
  } catch { /* */ }
  return 'zh';
}

let _lang = _detect();
let _t = bundles[_lang] || bundles.zh;

/** Current language code */
function getLang() { return _lang; }

/** Current translations bundle */
function getT() { return _t; }

/** Shorthand — same as getT() */
const t = new Proxy({}, {
  get(_, prop) { return _t[prop]; },
});

/**
 * Switch language at runtime.
 * @param {string} newLang  'zh' | 'en'
 * @param {boolean} [reload=true]  Reload the page after switch
 */
function setLang(newLang, reload = true) {
  if (newLang !== 'zh' && newLang !== 'en') return;
  _lang = newLang;
  _t = bundles[newLang] || bundles.zh;
  try { localStorage.setItem(STORAGE_KEY, newLang); } catch { /* */ }
  if (reload) location.reload();
}

/**
 * Create a language toggle widget. Returns the container element.
 * Call this after the page header is rendered.
 * @param {HTMLElement} [parent] — if provided, appends to it
 * @returns {HTMLElement}
 */
function createLangToggle(parent) {
  const wrap = document.createElement('div');
  wrap.className = 'lang-toggle';
  wrap.style.cssText = 'display:inline-flex;gap:2px;align-items:center;margin-left:12px;';

  const btnZh = document.createElement('button');
  btnZh.textContent = '中';
  btnZh.title = '切换到中文';
  btnZh.className = 'lang-btn' + (_lang === 'zh' ? ' active' : '');
  btnZh.onclick = () => setLang('zh');

  const btnEn = document.createElement('button');
  btnEn.textContent = 'En';
  btnEn.title = 'Switch to English';
  btnEn.className = 'lang-btn' + (_lang === 'en' ? ' active' : '');
  btnEn.onclick = () => setLang('en');

  const style = document.createElement('style');
  style.textContent = `
    .lang-btn {
      font-size: 12px; padding: 2px 8px; border-radius: 5px;
      border: 1px solid #ddd; background: transparent; color: #888;
      cursor: pointer; transition: all .15s; line-height: 1.4;
    }
    .lang-btn:hover { background: #f5f5f0; color: #333; }
    .lang-btn.active { background: #2d2d2d; color: #fff; border-color: #2d2d2d; }
  `;
  wrap.appendChild(style);
  wrap.appendChild(btnZh);
  wrap.appendChild(btnEn);

  if (parent) parent.appendChild(wrap);
  return wrap;
}

// Set window.I18n immediately so classic <script> blocks can use it.
if (typeof window !== 'undefined') {
  window.I18n = { getLang, getT, setLang, createLangToggle, t: _t, bundles };
}

