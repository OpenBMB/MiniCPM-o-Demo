# iOS 触控滚动修复（全站）

## 问题

在 iPhone / iPad 上访问各页面时，页面无法通过触控上下滑动，导致底部的控制按钮或输入区域不可见、无法操作。

## 根因分析

| 原因 | 说明 |
|------|------|
| `body { height: 100vh }` | 把 body 锁死在一屏高度，内容溢出时无法滚动。iOS Safari 的 `100vh` 还包含地址栏后面的区域，导致底部被遮挡 |
| `.col-left, .col-right { overflow: hidden }` | 列容器裁切溢出内容，触控事件无法触发滚动 |
| 媒体查询断点太小（768px） | iPad 竖屏 810px、横屏 1024px+，全部未命中 |

## 涉及页面与修改文件

| 页面 | 修改文件 | 布局类型 |
|------|----------|----------|
| Omni Full-Duplex | `static/omni/omni.css` | 两列 grid + 底部控制栏 |
| Audio Full-Duplex | `static/audio-duplex/audio-duplex.css` | 两列 grid + 底部控制栏 |
| Half-Duplex Audio | `static/half-duplex/half-duplex.css` | 两列 grid + 底部控制栏 |
| Turn-based Chat | `static/turnbased.html`（内联 style） | 单列 flex + 底部输入区 |

---

## 修复方案 A：Duplex 系列页面（Omni / Audio / Half-Duplex）

这三个页面共享 `duplex-shared.css` 的两列布局，各自的页面 CSS 中添加了相同的覆盖规则。

### 1. 全局允许滚动（无媒体查询）

```css
html {
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
}
body {
    height: auto !important;
    min-height: 100vh;
    overflow-y: auto !important;
    -webkit-overflow-scrolling: touch;
}
```

- 用 `!important` 覆盖 `duplex-shared.css` 的 `height: 100vh`
- `-webkit-overflow-scrolling: touch` 启用 iOS 惯性滚动
- 桌面端内容未溢出时不会出现滚动条，无副作用

### 2. 平板 & 手机：固定底部控制栏（≤1024px）

```css
@media (max-width: 1024px) {
    .main          { height: auto; overflow: visible; padding-bottom: 72px; }
    .col-left,
    .col-right     { overflow: visible; }
    .panel-controls {
        position: fixed; bottom: 0; left: 0; right: 0;
        z-index: 50;
        border-radius: 10px 10px 0 0;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.1);
    }
}
```

- 断点提升到 1024px，覆盖所有 iPad 尺寸
- `.panel-controls` 固定在屏幕底部，始终可见
- `padding-bottom: 72px` 防止页面内容被固定栏遮挡

### 3. 触控设备兜底（hover: none + pointer: coarse）

```css
@media (hover: none) and (pointer: coarse) {
    /* 与上面相同的规则 */
}
```

- 即使 iPad Pro 横屏宽度 > 1024px，该查询也能命中触控设备

---

## 修复方案 B：Turn-based Chat 页面

该页面使用内联 CSS，布局与 Duplex 系列不同（ChatGPT 风格单列布局）。

### 1. body 高度改为 min-height

```css
html {
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
}
body {
    min-height: 100vh;
    min-height: 100dvh;  /* 动态视口高度，解决 iOS 地址栏遮挡 */
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
}
```

- `100dvh`（Dynamic Viewport Height）在 iOS Safari 15.4+ 正确反映不含地址栏的实际可见高度
- `100vh` 作为不支持 `dvh` 的浏览器的 fallback

### 2. 初始视图允许滚动

```css
.initial-view {
    overflow-y: auto;
    -webkit-overflow-scrolling: touch;
}
```

- 初始视图包含系统配置卡片 + 输入区，在小屏设备上容易溢出

### 3. 对话视图 flex 收缩

```css
.chat-view {
    min-height: 0;  /* 确保 flex 子元素在父容器缩小时正确收缩 */
}
```

---

## 断点层级总结（Duplex 系列）

| 断点 | 覆盖设备 | 效果 |
|------|----------|------|
| 无限制 | 所有设备 | body 允许滚动 |
| ≤ 1024px | iPad 竖屏、所有手机 | 固定底部控制栏 + 内容区可滑动 |
| hover:none + pointer:coarse | 所有触控设备 | 同上（兜底） |
| ≤ 768px | 手机 | 单列布局 + 全屏模式优化 |
| ≤ 480px | 小屏手机 | 更紧凑的控制栏 |

## 注意事项

- Duplex 系列的修改仅在各自的页面 CSS 中，不修改共享的 `duplex-shared.css`，避免影响未来新增页面
- Turn-based 页面的修改在内联 `<style>` 中
- 全屏模式（video fullscreen）下控制栏有独立的固定定位逻辑，不受此修改影响
