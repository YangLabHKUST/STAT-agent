---
id: analyzer
version: 1
role: analyzer
description: Analyzer - summary, data changes, optional interpretation, execution-issue verdict.
source: stat_agent/agent/spatial_agent_core.py:2733-2764 (_analyze_execution_results, STAT 0.1.27), text unchanged
---
Analyze spatial transcriptomics code execution results.

{% if session_info %}**{{ session_info }}**
{% endif %}{% if history_block %}**Conversation History:**
{{ history_block }}
**[End of Conversation History]**
{% endif %}
User asked: "{{ user_query }}"

{{ assembled_context }}

Provide JSON response:
{
  "response_summary": "Brief summary of entire response in under 4 sentences (what was done and key results)",
  "data_changes": "Detailed description of what was added/changed in session data (columns, obsm keys, uns keys, ROIs). Be specific about names, types, and values. 2-3 sentences.",
  "interpretation": {% if has_interpret %}'{{ interpret_instruction }}' - Provide biological interpretation in 2-4 sentences{% else %}null{% endif %},
  "key_findings": {% if has_interpret %}["List of 1-3 specific biological findings with numbers"]{% else %}[]{% endif %},
  "execution_issues": {
    "has_issues": true/false,
    "issue_type": "error" | "validation_failed" | "partial_success" | "no_effect" | null,
    "explanation": "Clear explanation of what went wrong and why" (only if has_issues=true)
  }
}

{% raw %}Rules:
- response_summary: Always provide, covers entire response
- data_changes: Always provide, be detailed and specific
- interpretation: Only if requested above (otherwise null)
- key_findings: Only if interpretation requested (otherwise empty array)
- execution_issues: Detect if code failed or didn't achieve its goal
  - has_issues=true ONLY if: Python error occurred, validation failed, code had no effect, or the user's goal was NOT achieved
  - issue_type: "error" (Python exception), "validation_failed" (data validation failed), "partial_success" (some parts worked but goal incomplete), "no_effect" (code ran but changed nothing)
  - explanation: User-friendly explanation of what went wrong and why
  - IMPORTANT: Truncated print output is NOT an issue. Output is truncated at ~3000 chars for display purposes only — the actual data in memory is complete. Do NOT flag truncated output as partial_success if the code executed successfully and the data was correctly modified/analyzed.

Output valid JSON only:{% endraw %}
