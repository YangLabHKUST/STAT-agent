from matplotlib.colors import to_rgba_array
from tests.fixtures.sessions import make_adata, make_session

from stat_agent.core.colors import (
    extract_existing_colors,
    generate_distinct_colors,
    normalize_color,
    normalize_palette,
)
from stat_agent.io.session_store import load_session, save_session


def test_normalize_formats():
    assert normalize_color("#ff0000") == "#ff0000"
    assert normalize_color("#F00") == "#ff0000"
    assert normalize_color("rgb(1, 2, 3)") == "#010203"
    assert normalize_color("rgba(255,0,16,0.5)") == "#ff0010"
    assert normalize_color("red") == "#ff0000"
    assert normalize_color("rgb(300, 0, 0)") is None and normalize_color("not a colour") is None
    assert normalize_color(3) is None
    assert normalize_palette({"a": "rgb(0, 0, 0)", "b": "nope"}) == {"a": "#000000"}


def test_generation_is_deterministic_and_keeps_existing():
    a = generate_distinct_colors(["b", "a", "c"])
    b = generate_distinct_colors(["c", "a", "b"])
    assert a == b and len(a) == 3
    c = generate_distinct_colors(["a", "b", "c"], existing={"a": "rgb(0, 0, 0)"})
    assert c["a"] == "#000000"
    assert all(v.startswith("#") and len(v) == 7 for v in c.values())


def test_palette_is_what_matplotlib_reads():
    """Generated code colours its plots with the palette in adata.uns (the demo's "what is the data like?" coloured
    a scatter of every cell with it); matplotlib rejected the CSS rgb() strings STAT used to store."""
    session = make_session("S1_cell_gene")
    palette = session.get_slice(0).get_celltype_colors()
    obs = session.get_slice(0).adata.obs
    assert to_rgba_array([palette[ct] for ct in obs["celltype"]]).shape == (len(obs), 4)


def test_extract_existing_scanpy_list_format():
    adata = make_adata(n_obs=50, celltypes=["x", "y"])
    adata.uns["celltype_colors"] = ["#ff0000", "#00ff00"]
    cats = list(adata.obs["celltype"].cat.categories)
    out = extract_existing_colors(adata)
    assert out[cats[0]] == "#ff0000" and out[cats[1]] == "#00ff00"


def test_saved_session_with_css_colours_loads_as_hex(tmp_path):
    session = make_session("S1_cell_gene")
    old = {ct: "rgb(176, 222, 59)" for ct in session.get_slice(0).celltypes}
    session.celltype_colors = dict(old)
    session.get_slice(0).set_celltype_colors(old)
    loaded, _ = load_session(save_session(session, target=tmp_path / "saved"))
    assert set(loaded.celltype_colors.values()) == {"#b0de3b"}
    assert loaded.get_slice(0).get_celltype_colors() == {ct: "#b0de3b" for ct in old}
