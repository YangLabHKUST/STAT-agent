"""``stat-bench`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_convert(args) -> int:
    from stat_bench.suite import convert_legacy_queries

    suite = convert_legacy_queries(args.queries_json, suite_id=args.suite_id)
    p = suite.save(args.out)
    print(f"wrote {p} ({len(suite.queries)} queries)")
    return 0


def cmd_run(args) -> int:
    from stat_bench.runner import plan_cells, run
    from stat_bench.suite import Suite
    from stat_bench.systems.stat import StatSystem

    suite = Suite.load(args.suite)
    data_root = args.data_root or __import__("os").environ.get(suite.data_root_env)
    cells = plan_cells(suite, system=args.system, driver=args.driver, ablation=args.ablation, policy=args.policy, backbone=args.backbone, seeds=args.seeds, out=Path(args.out), data_root=data_root, ids=args.queries.split(",") if args.queries else None, categories=args.category.split(",") if args.category else None, execute=not args.no_execute)
    if args.dry_run:
        for c in cells:
            print(c.query.id, c.seed_index, c.out_dir)
        return 0
    if args.system != "stat":
        print(f"system {args.system!r} is not wired into stat-bench yet (adapters for vanilla/biomni/spatialagent are planned)", file=sys.stderr)
        return 2
    results = run(StatSystem(), cells, resume=not args.no_resume, retry_failed=args.retry_failed, manifest_path=Path(args.out) / "manifest.json", on_result=lambda r: print(f"{r.query_id} seed{r.seed_index}: {r.exit_status} ({r.wall_s:.0f}s, {r.total_tokens} tokens)"))
    ok = sum(1 for r in results if r.success)
    print(f"{ok}/{len(results)} succeeded")
    return 0


def cmd_summarize(args) -> int:
    from stat_bench.summarize import load_results, write_summary

    results = []
    for d in args.results:
        results += load_results(d)
    p = write_summary(results, Path(args.out) / "summary.json", by=tuple(args.by.split(",")))
    print(p.with_suffix(".md").read_text())
    return 0


def cmd_replay(args) -> int:
    from stat_bench.replay import report, save_report, trace_decisions

    old = trace_decisions([Path(d) for d in args.logs])
    new = json.loads(Path(args.new).read_text()) if args.new else []
    expected = json.loads(Path(args.expected).read_text()) if args.expected else {}
    rep = report(old, new, expected)
    save_report(rep, Path(args.out))
    print(f"{rep['n_compared']} compared, gate={rep['gate']}, unexplained={rep['unexplained']}")
    (Path(args.out) / "old_decisions.json").write_text(json.dumps(old, indent=1), encoding="utf-8")
    return 0 if rep["gate"] == "pass" or not new else 1


def cmd_corrupt(args) -> int:
    from stat_bench.corrupt import corrupt_registry

    manifest = corrupt_registry(Path(args.skills), Path(args.out), severity=args.severity, kinds=tuple(args.kinds.split(",")), n=args.n, seed=args.seed)
    print(f"{len(manifest)} skills corrupted -> {args.out}/manifest.json")
    return 0


def cmd_session(args) -> int:
    from stat_bench.session_bench import SessionScript

    scripts = [SessionScript.load(p) for p in (sorted(Path(args.script).glob("*.yaml")) if Path(args.script).is_dir() else [Path(args.script)])]
    print(f"{len(scripts)} script(s); real-LLM session runs need a dataset per fixture - use the Python API (stat_bench.session_bench.run_script) with an agent_factory", file=sys.stderr)
    return 0


def cmd_judge(args) -> int:
    from stat_bench.judge import judge_results

    v = judge_results(args.results, args.judge_model, out=args.out)
    print(f"{len(v)} verdicts -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stat-bench", description="STAT benchmark harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="convert STAT 0.1 queries.json into a suite file")
    c.add_argument("queries_json")
    c.add_argument("--out", required=True)
    c.add_argument("--suite-id", default="stat40")
    c.set_defaults(func=cmd_convert)
    r = sub.add_parser("run", help="run a suite")
    r.add_argument("--suite", required=True)
    r.add_argument("--system", default="stat")
    r.add_argument("--driver", choices=["staged", "loop"], default="staged")
    r.add_argument("--ablation", default="A0")
    r.add_argument("--policy", choices=["scripted", "auto", "interactive"], default="scripted")
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--backbone", default="openai/gpt-4o")
    r.add_argument("--out", required=True)
    r.add_argument("--data-root")
    r.add_argument("--queries")
    r.add_argument("--category")
    r.add_argument("--no-execute", action="store_true")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summarize", help="aggregate results with bootstrap CIs")
    s.add_argument("results", nargs="+")
    s.add_argument("--out", required=True)
    s.add_argument("--by", default="system,driver,policy")
    s.set_defaults(func=cmd_summarize)
    rp = sub.add_parser("replay", help="decision-level replay report against STAT 0.1 logs")
    rp.add_argument("--logs", nargs="+", required=True)
    rp.add_argument("--new", help="JSON with new decision records (same shape as old_decisions.json)")
    rp.add_argument("--expected", help="expected_divergences.json")
    rp.add_argument("--out", required=True)
    rp.set_defaults(func=cmd_replay)
    co = sub.add_parser("corrupt", help="corrupt skill frontmatter for the containment experiment")
    co.add_argument("--skills", required=True)
    co.add_argument("--out", required=True)
    co.add_argument("--severity", type=int, default=1)
    co.add_argument("--kinds", default="data_levels,modalities,num_slices,prereq,description")
    co.add_argument("--n", type=int, default=6)
    co.add_argument("--seed", type=int, default=0)
    co.set_defaults(func=cmd_corrupt)
    se = sub.add_parser("session", help="SessionBench scripts")
    se.add_argument("script")
    se.set_defaults(func=cmd_session)
    j = sub.add_parser("judge", help="LLM judge over a results directory")
    j.add_argument("--results", required=True)
    j.add_argument("--judge-model", required=True)
    j.add_argument("--out", required=True)
    j.set_defaults(func=cmd_judge)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
