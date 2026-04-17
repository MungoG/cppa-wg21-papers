---
title: "Mermaid Test"
document: P9998R0
date: 2026-01-01
reply-to:
  - "Test Author <test@example.com>"
---

## Flowchart

```mermaid
flowchart TD
    A[Start] --> B{Decision}
    B -->|yes| C[Process]
    B -.->|no| D((End))
    C --> D
```
