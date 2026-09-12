---
id: generator_user
version: 1
role: generator
description: Code-generator user message - session context, memory, skill guidance, query.
source: stat_agent/agent/spatial_agent_core.py:1818-1984 (_build_prompt, STAT 0.1.27); section texts unchanged, blocks assembled by roles/context.py
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

{% endif %}{% if skill_guidance %}================================================================================
**IMPORTANT: SKILL GUIDANCE PROVIDED**
================================================================================

You have been provided with specialized skill guidance below.
Follow these instructions EXACTLY to complete the user's request.
The skill provides a proven workflow - use it step-by-step.

{{ skill_guidance }}

================================================================================

{% endif %}**User Query:** {{ user_query }}

Please provide a helpful response. If code is needed, wrap it in ```python code fences.
