from tests.fixtures.sessions import make_adata

from stat_agent.core.colors import extract_existing_colors, generate_distinct_colors, normalize_color_to_rgb


def test_normalize_formats():
    assert normalize_color_to_rgb("#ff0000") == "rgb(255, 0, 0)"
    assert normalize_color_to_rgb("#f00") == "rgb(255, 0, 0)"
    assert normalize_color_to_rgb("rgb(1, 2, 3)") == "rgb(1, 2, 3)"
    assert normalize_color_to_rgb("red") == "rgb(255, 0, 0)"
    assert normalize_color_to_rgb(3) is None


def test_generation_is_deterministic_and_keeps_existing():
    a = generate_distinct_colors(["b", "a", "c"])
    b = generate_distinct_colors(["c", "a", "b"])
    assert a == b and len(a) == 3
    c = generate_distinct_colors(["a", "b", "c"], existing={"a": "#000000"})
    assert c["a"] == "rgb(0, 0, 0)"
    assert all(v.startswith("rgb(") for v in c.values())


def test_extract_existing_scanpy_list_format():
    adata = make_adata(n_obs=50, celltypes=["x", "y"])
    adata.uns["celltype_colors"] = ["#ff0000", "#00ff00"]
    cats = list(adata.obs["celltype"].cat.categories)
    out = extract_existing_colors(adata)
    assert out[cats[0]] == "rgb(255, 0, 0)" and out[cats[1]] == "rgb(0, 255, 0)"
