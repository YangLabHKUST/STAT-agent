---
id: summarizer
version: 1
role: summarizer
description: Rolling conversation summary regenerated from the previous summary plus evicted turns' handoffs.
source: new in v2 (STAT 0.1 built the summary by string concatenation); one-sentence style borrowed from memory.py:609-631
---
You maintain a running summary of a spatial transcriptomics analysis session.

**Previous summary (may be empty):**
{{ previous_summary }}

**Turns that are being compressed (oldest first):**
{{ evicted_turns_block }}

Write an updated summary that a colleague could read to understand what has been done so far.

REQUIREMENTS:
- At most 8 sentences, past tense, specific about column/key/ROI names and counts.
- Keep every user-stated fact that matters later (species, tissue, reference paths, parameter choices).
- Keep unresolved issues and failed attempts.
- Do not repeat the previous summary verbatim; merge it.

Respond with JSON only:
{"summary": "...", "facts": [{"key": "species", "value": "human", "confidence": 0.9}]}
