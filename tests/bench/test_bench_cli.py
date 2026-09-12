from stat_bench.cli import build_parser


def test_cli_parses():
    p = build_parser()
    a = p.parse_args(["run", "--suite", "s.yaml", "--out", "o", "--seeds", "3", "--ablation", "A2", "--driver", "loop"])
    assert a.seeds == 3 and a.ablation == "A2" and a.driver == "loop"
    assert p.parse_args(["convert", "q.json", "--out", "s.yaml"]).suite_id == "stat40"
