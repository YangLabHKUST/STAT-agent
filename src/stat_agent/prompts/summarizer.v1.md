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
- At most {{ max_words }} words (the budget grows with the number of turns the summary covers): add sentences rather than dropping earlier work.
- Chronological and past tense, specific about column/key/ROI names, parameters and counts. Number the analyses in the order they were done. When a later turn re-did, replaced or deleted an earlier result, keep the earlier one and mark it "(superseded by ...)" or "(deleted at ...)" instead of dropping it: the user may ask what was done the first time.
- Keep every standing instruction the user gave (for example "do not overwrite the celltype column", "leave slice 2 alone") in the user's words, and every user-stated fact that matters later (species, tissue, reference paths, parameter choices).
- Keep unresolved issues and failed attempts.
- Do not repeat the previous summary verbatim; merge it, but never lose an item it contains.

Respond with JSON only:
{"summary": "...", "facts": [{"key": "species", "value": "human", "confidence": 0.9}, {"key": "instruction: celltype", "value": "do not overwrite the celltype column", "confidence": 1.0}]}
Every standing user instruction goes into "facts" with a key that starts with "instruction:".
