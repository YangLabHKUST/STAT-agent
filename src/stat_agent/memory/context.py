"""Context assembly: the text blocks each role receives.

These functions reproduce the STAT 0.1 formatting byte-for-byte where a v1 prompt is in
use (so the replay report is meaningful) and are the single place to change when a prompt
version needs more or different context.
"""

from __future__ import annotations

from collections.abc import Iterable

from stat_agent.core.view import InterfaceState, SessionSummary, SliceSummary
from stat_agent.memory.state import ConversationMemory


def planner_slice_line(s: SliceSummary) -> str:
    desc = f"  - Slice {s.slice_id}: {s.tissue_name} ({s.modality}, {s.data_level}-level, {s.n_obs:,} observations, {s.n_vars:,} genes"
    if s.has_celltype and s.celltypes:
        desc += f", {len(s.celltypes)} cell types"
    if s.image_names:
        desc += f", images: {', '.join(s.image_names)}"
    return desc + ")"


def planner_session_block(summary: SessionSummary) -> str:
    lines = [f"- Total slices: {summary.n_slices}"] + [planner_slice_line(s) for s in summary.slices]
    if summary.rois:
        lines.append("")
        lines.append("Available ROIs:")
        for r in summary.rois:
            lines.append(f"  - '{r.name}' on slice {r.slice_id} ({r.n_obs:,} cells)")
    return "\n".join(lines)


def history_block(memory: ConversationMemory | None, style: str = "planner") -> str:
    if memory is None:
        return ""
    text = memory.history_text()
    if not text:
        return ""
    if style == "planner":
        return f"\n**Conversation History (what has been done so far):**\n{text}\n**[End of Conversation History]**\n"
    if style == "verifier":
        return f"\n**Conversation History (what has been done so far):**\n{text}\n"
    return text


def clarifications_block(answers: Iterable[tuple[str, str]]) -> str:
    pairs = list(answers)
    if not pairs:
        return ""
    out = "\n**Previous Clarifications of Query:**\n"
    for i, (q, a) in enumerate(pairs, 1):
        out += f"{i}. Q: {q}\n   A: {a}\n"
    return out


def interface_block(interface: InterfaceState | None) -> str:
    if interface is None:
        return ""
    desc = interface.describe()
    return f"\n**Interface state (what the user is currently looking at):** {desc}\n" if desc else ""


def verifier_slice_line(s: SliceSummary, stale: Iterable[str] = ()) -> str:
    desc = f"  - Slice {s.slice_id}: {s.tissue_name} ({s.modality}, {s.data_level}-level, {s.n_obs:,} observations)"
    if s.has_celltype and s.celltypes:
        desc += f", has celltype annotations ({len(s.celltypes)} types)"
    else:
        desc += ", NO celltype annotations"
    if s.image_names:
        desc += f", images: {', '.join(s.image_names)}"
    if s.obs_columns:
        cols = s.obs_columns[:10]
        desc += f"\n    Available columns: {', '.join(cols)}"
        if len(s.obs_columns) > 10:
            desc += f" ... ({len(s.obs_columns)} total)"
    if s.obsm_keys:
        desc += f"\n    obsm keys: {', '.join(s.obsm_keys[:10])}"
    stale_here = [a for a in stale if a.startswith(f"s{s.slice_id}:")]
    if stale_here:
        desc += f"\n    STALE (upstream changed, must be recomputed before use): {', '.join(stale_here)}"
    return desc


def verifier_session_block(summary: SessionSummary, target_slice_ids: Iterable[int], stale: Iterable[str] = ()) -> str:
    targets = list(target_slice_ids)
    lines = [f"- Total slices: {summary.n_slices}"]
    for sid in targets:
        s = summary.slice(sid)
        if s is not None:
            lines.append(verifier_slice_line(s, stale))
    rois = summary.rois_on(targets)
    if rois:
        lines.append("")
        lines.append("Available ROIs on target slices:")
        for r in rois:
            lines.append(f"  - '{r.name}' on slice {r.slice_id} ({r.n_obs:,} cells)")
    return "\n".join(lines)


def user_responses_block(answers: Iterable[tuple[str, str]]) -> str:
    pairs = list(answers)
    if not pairs:
        return ""
    out = "\n**Previously Collected Information:**\n"
    for q, a in pairs:
        out += f"  Q: {q}\n  A: {a}\n"
    return out


def generator_session_block(summary: SessionSummary, roi_names: list[str]) -> str:
    """The STAT 0.1 ``**Current Session:**`` block of ``_build_prompt`` (text unchanged)."""
    n_slices = summary.n_slices
    has_protein = "protein" in summary.modalities
    if n_slices > 1 and has_protein:
        data_format = "multi_slice_multi_omics"
    elif n_slices > 1:
        data_format = "multi_slice"
    elif has_protein:
        data_format = "multi_omics"
    else:
        data_format = "single_slice"
    parts: list[str] = []
    if data_format == "single_slice":
        parts.append("- Data Format: Single slice")
    elif data_format == "multi_slice":
        parts.append(f"- Data Format: Multi-slice ({n_slices} slices)")
    elif data_format == "multi_omics":
        parts.append("- Data Format: Multi-omics (gene + protein)")
    else:
        parts.append(f"- Data Format: Multi-slice, multi-omics ({n_slices} slices, gene + protein)")
    total_cells = sum(s.n_obs for s in summary.slices)
    parts.append(f"- Cells/spots: {total_cells:,}")
    if roi_names:
        parts.append(f"- Available ROIs: {', '.join(roi_names)}")
    else:
        parts.append("- ROIs: None defined yet")
    if data_format in ("multi_slice", "multi_slice_multi_omics") and summary.slices:
        parts.append("")
        parts.append("**Available Slices:**")
        for s in summary.slices:
            unit = "genes" if s.modality == "gene" else "proteins"
            parts.append(f"- Slice {s.slice_id}: {s.tissue_name} ({s.n_obs:,} observations, {s.modality}, {s.data_level}-level, {s.n_vars:,} {unit})")
            parts.append(f"  Access: session.get_slice({s.slice_id}).adata")
            if s.celltypes:
                parts.append(f"  Cell types: {', '.join(s.celltypes)}")
    elif data_format == "single_slice" and summary.slices:
        s = summary.slices[0]
        parts.append(f"- Data level: {s.data_level}")
        parts.append(f"- Genes/features: {s.n_vars:,}")
        if s.celltypes:
            parts.append(f"- Cell types: {', '.join(s.celltypes)}")
    if data_format == "multi_omics":
        parts.append("")
        parts.append("**Multi-Omics Data:**")
        gene = [s for s in summary.slices if s.modality == "gene"]
        prot = [s for s in summary.slices if s.modality == "protein"]
        if gene:
            parts.append(f"- Gene data: {gene[0].n_vars} features, {gene[0].n_obs} cells")
            parts.append(f"  Access: session.get_slice({gene[0].slice_id}).adata")
        if prot:
            parts.append(f"- Protein data: {prot[0].n_vars} features, {prot[0].n_obs} cells")
            parts.append(f"  Access: session.get_slice({prot[0].slice_id}).adata")
    return "\n".join(parts)


def analyzer_session_info(summary: SessionSummary, roi_names: list[str]) -> str:
    if not summary.has_data:
        return ""
    total = sum(s.n_obs for s in summary.slices)
    info = f"Session: {summary.n_slices} slice(s), {total:,} cells"
    if roi_names:
        info += f", ROIs: {', '.join(roi_names)}"
    for s in summary.slices:
        info += f"\nSlice {s.slice_id} ({s.modality}, {s.data_level}): {s.n_vars:,} genes"
        if s.celltypes:
            info += f", celltypes = {', '.join(s.celltypes)}"
    return info


def skill_guidance_block(name: str, body: str) -> str:
    return f"**Skill: {name}**\n{body}"


def plan_block(original_query: str, step_number: int, total_steps: int, description: str) -> str:
    bar = "=" * 80
    return (
        f"{bar}\n**MULTI-STEP PLAN EXECUTION**\n{bar}\n\n**Original Task:** {original_query}\n\n**Current Step:** [Step {step_number} of {total_steps}] {description}\n\n"
        "**Complete this step fully in this response.** If you need to inspect the data first, do it inside the same code block and continue with the analysis; "
        "do not stop after an inspection and do not defer the step's work to a later step or turn.\n\n"
        f"{bar}\n"
    )


__all__ = [
    "planner_session_block",
    "history_block",
    "clarifications_block",
    "interface_block",
    "verifier_session_block",
    "user_responses_block",
    "generator_session_block",
    "analyzer_session_info",
    "skill_guidance_block",
    "plan_block",
]
