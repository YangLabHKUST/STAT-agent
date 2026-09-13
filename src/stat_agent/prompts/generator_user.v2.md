---
id: generator_user
version: 2
role: generator
description: Code-generator user message - session context, memory, skill guidance, query.
source: "generator_user.v1 plus: finish the requested work in this response; the continuation block; the completed steps of this turn; the Python namespace listing"
---
{% if plan_block %}{{ plan_block }}
{% endif %}{% if session_block %}**Current Session:**
{{ session_block }}
{% endif %}{% if entity_block %}{{ entity_block }}
{% endif %}{% if not has_data %}**Note:** No data loaded yet. Guide the user to load data first.

{% endif %}{% if state_block %}**Session Artefacts (provenance; STALE items must be recomputed before use):**
{{ state_block }}

{% endif %}{% if history_block %}**Conversation History:**
{{ history_block }}
**[End of Conversation History]**

{% endif %}{% if steps_block %}{{ steps_block }}

{% endif %}{% if namespace_block %}{{ namespace_block }}

{% endif %}{% if unguarded_note %}{{ unguarded_note }}

{% endif %}{% if skill_guidance %}================================================================================
**IMPORTANT: SKILL GUIDANCE PROVIDED**
================================================================================

You have been provided with specialized skill guidance below.
Follow these instructions EXACTLY to complete the user's request.
The skill provides a proven workflow: run all of its stages in this response, as consecutive code blocks, and do not stop after a preparation or inspection stage.

{{ skill_guidance }}

================================================================================

{% endif %}{% if continuation_block %}{{ continuation_block }}

{% endif %}**User Query:** {{ user_query }}

Please provide a helpful response. If code is needed, wrap it in ```python code fences.
Do the complete task in this response: any exploratory check (value counts, column inspection) belongs in the same code block as the analysis it informs. Do not stop after inspecting and promise the analysis for later.
