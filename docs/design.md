# 技术 & 设计规范

## LabelStudio 标注界面设计

### 布局结构

```
┌──────────────────────────────────────┐
│         对话文本标注 (Header)          │
├─────────────────┬────────────────────┤
│  用户消息 (user)  │  助手回复 (assistant)│
│  (蓝色背景卡片)   │  (绿色背景卡片)     │
├─────────────────┴────────────────────┤
│         质量判断 (Choices)             │
│    ○ 有用    ○ 无用                   │
├──────────────────────────────────────┤
│  错误类型（可选文本框）                │
│  ┌────────────────────────────────┐  │
│  │ placeholder: 事实错误、逻辑矛盾..│  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

### XML 配置要点

- 使用 `<View style="display:flex">` 实现左右分栏
- `<Text>` 绑定 `$user` / `$assistant` 数据字段
- `<Choices choice="single-radio" required="true">` 强制二选一
- `<TextArea>` 不设置 `required`，为可选字段

### 数据流

```
sample_conversations.json
        │
        ▼  (Web UI Import)
  LabelStudio Project
        │
        ▼  (人工标注)
  Annotations (SQLite)
        │
        ▼  (Export API / SDK)
  annotations_export.json
```

## 关键设计决策

1. **错误类型不设为 required**：标注者可能无法判断具体错误类型，但能判断有用/无用
2. **使用 single-radio 而非 checkbox**：每条对话只能是有用或无用，互斥
3. **错误类型用 TextArea 而非 Choices**：允许标注者自由输入，超越预设分类
