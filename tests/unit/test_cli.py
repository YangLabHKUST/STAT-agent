from stat_agent.cli.main import build_parser


def test_cli_parses_subcommands():
    p = build_parser()
    a = p.parse_args(["skill", "validate", "--strict"])
    assert a.cmd == "skill" and a.skill_cmd == "validate" and a.strict
    b = p.parse_args(["chat", "/data", "-m", "hello", "--mode", "auto"])
    assert b.cmd == "chat" and b.message == "hello" and b.mode == "auto"
    c = p.parse_args(["web", "--port", "9000"])
    assert c.port == 9000
