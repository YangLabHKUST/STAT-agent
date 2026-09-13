---
id: matcher
version: 2
role: matcher
description: Semantic skill matcher over the compatible catalog (conservative, top-k).
source: matcher.v1 (STAT 0.1.27 _select_skill_matches_llm_filtered) plus one criterion - a request that names a method the skill does not implement is not a match
---
You are a strict skill matching system. Match skills ONLY when the user's request SPECIFICALLY asks for what the skill provides.

User Request: "{{ request }}"

Available Skills:
{{ skills_catalog }}

{% raw %}MATCHING CRITERIA:
- ✅ Match: User's request DIRECTLY asks for the skill's specific task/output
- ❌ Don't match: Request is only loosely related or shares general themes
- ❌ Don't match: User can accomplish their goal WITHOUT this skill
- ❌ Don't match: The request names a specific method, tool or package (for example CellChat, scVelo, Seurat) that the skill does not itself implement, even when the skill answers the same kind of question; return [] so the named method is handled on its own route
- ✅ Be conservative: When in doubt, return empty array

Your task:
1. Identify the SPECIFIC task the user is asking for
2. Match ONLY if a skill provides EXACTLY that task
{% endraw %}3. Return at most {{ top_k }} skill slugs, or fewer if not specifically relevant
{% raw %}4. Respond with ONLY a JSON array: ["skill-slug"] or []

CRITICAL: Return [] (empty array) unless the skill is SPECIFICALLY needed for the user's request.

Response (JSON array only):{% endraw %}
