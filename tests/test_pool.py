from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "external/starter-kit/smart_buildings"
STATIONS = ("fairbanks", "denver", "atlanta", "phoenix", "singapore")


def _training_strips():
    return {p["weather"] for p in yaml.safe_load((KIT / "training_set.yaml").read_text())["scenarios"]}


def test_pool_sizes():
    from harness.gen_pool import load_split
    sizes = {s: len(load_split(s)) for s in ("pool_train", "pool_test", "minneapolis", "large")}
    assert sizes == {"pool_train": 20, "pool_test": 20, "minneapolis": 5, "large": 5}
    seen = set()
    for s in ("pool_train", "pool_test", "minneapolis", "large"):
        dirs = {str(p) for p in load_split(s)}
        assert not dirs & seen
        seen |= dirs
        assert all((Path(p) / "config.gin").exists() for p in dirs)


def test_pool_disjoint_from_training():
    from harness.gen_pool import load_split
    train_strips = _training_strips()
    for s in ("pool_train", "pool_test", "minneapolis", "large"):
        assert not any(p.parent.name.startswith("building_train") for p in load_split(s))
    test_strips = {p.name for p in load_split("pool_test")}
    assert not test_strips & train_strips  # held-out generated set: unseen buildings AND unseen weather
    assert not test_strips & {p.name for p in load_split("pool_train")}
    assert {w.split("_")[0] for w in test_strips} == set(STATIONS)
    for s in ("pool_train", "pool_test"):
        per = [sum(p.name.startswith(st) for p in load_split(s)) for st in STATIONS]
        assert per == [4, 4, 4, 4, 4]


def test_pool_ranges():
    from harness.gen_pool import load_split
    for s in ("pool_train", "pool_test", "minneapolis"):
        for p in load_split(s):
            y = yaml.safe_load((ROOT / "pool/buildings" / f"{p.parent.name}.yaml").read_text())["floor_plan"]
            assert 318 <= y["width"] <= 430 and 188 <= y["height"] <= 301 and 2 <= y["num_floors"] <= 4


def test_folds():
    from harness.gen_pool import load_split
    shipped = {str(p) for p in load_split("shipped")}
    assert len(shipped) == 30
    for st in STATIONS:
        test = load_split(f"loso:{st}:test")
        train = {str(p) for p in load_split(f"loso:{st}:train")}
        assert len(test) == 6 and all(p.name.startswith(st) for p in test)
        assert train | {str(p) for p in test} == shipped and len(train) == 24
    sub = load_split("subset10")
    assert len(sub) == 10 and sorted(p.name.split("_")[0] for p in sub) == sorted(STATIONS * 2)
