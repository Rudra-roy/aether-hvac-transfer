"""Generated building pool + fold definitions.

    uv run python -m harness.gen_pool [--workers 8]

Writes pool/buildings/*.yaml, pool/scenarios/<building>/<weather>/, pool/pool.yaml and harness/folds.yaml.

Splits (each building appears once, with one weather strip):
  pool_train  20  shipped size ranges, weather strips from training_set.yaml (seen weather)  -> prior, tuning
  pool_test   20  shipped size ranges, strips NOT in training_set.yaml (unseen weather)      -> primary metric
  minneapolis  5  shipped size ranges, minneapolis strips (unseen station)
  large        5  the kit generator's DEFAULT_RANGES (bigger footprints), unseen strips    -> robustness/runtime
Each slot tries seeds 1000 + 100*slot + attempt until the kit accepts the floor plan, so the pool
is deterministic regardless of worker order.
"""

import argparse
import hashlib
import os
import random
import shutil
from multiprocessing import Pool
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "external/starter-kit"
KSB = KIT / "smart_buildings"
POOL = ROOT / "pool"
STATIONS = ("fairbanks", "denver", "atlanta", "phoenix", "singapore")
SHIPPED_RANGES = dict(width=(318, 430), height=(188, 301), num_floors=(2, 4), min_room=(25, 35), span=(11, 16), occ=(1, 2))


def training_pairs() -> list[tuple[str, str]]:
    return [(p["building"], p["weather"]) for p in yaml.safe_load((KSB / "training_set.yaml").read_text())["scenarios"]]


def strips(station: str) -> list[str]:
    return sorted((p.stem for p in (KSB / "weather").glob(f"{station}_*.yaml")), key=lambda s: int(s.split("_")[1]))


def slots() -> list[tuple[str, str, str]]:
    """(split, weather strip, size-range kind) for every building, in a fixed order."""
    used = {w for _, w in training_pairs()}
    out = []
    for st in STATIONS:
        seen = [w for w in strips(st) if w in used]
        unseen = [w for w in strips(st) if w not in used]
        out += [("pool_train", seen[i], "shipped") for i in range(4)]
        out += [("pool_test", unseen[i % len(unseen)], "shipped") for i in range(4)]
        out += [("large", unseen[0], "default")]
    out += [("minneapolis", w, "shipped") for w in strips("minneapolis")[:5]]
    return out


def _materials(n: int, tag: bytes = b"aether-pool-v1") -> list[dict]:
    from smart_buildings.smart_control.utils.materials import draw_materials
    digest = hashlib.sha256(tag).digest()
    return draw_materials(random.Random(int.from_bytes(digest[:8], "big")), n)


def _make(job) -> tuple[int, str, str, str]:
    slot, split, weather, kind, mats = job
    os.chdir(KIT)  # the kit resolves weather csvs relative to its root
    import sys
    sys.path.insert(0, str(KSB))
    from run_floor_generator import build_config
    from smart_buildings.smart_control.utils.scenario_generator import generate_scenario_from_config, load_scenario_from_parts

    for attempt in range(100):
        seed = 1000 + 100 * slot + attempt
        name = f"building_{seed}"
        cfg = build_config(seed, name, "composite", mats)
        if kind == "shipped":
            r = random.Random(seed)
            fp = cfg["floor_plan"]
            fp["width"] = r.randint(*SHIPPED_RANGES["width"])
            fp["height"] = r.randint(*SHIPPED_RANGES["height"])
            fp["num_floors"] = r.randint(*SHIPPED_RANGES["num_floors"])
            fp["min_room_size"] = r.randint(*SHIPPED_RANGES["min_room"])
            fp["max_room_size"] = fp["min_room_size"] + r.randint(*SHIPPED_RANGES["span"])
            cfg["simulation"]["occupants_per_zone"] = r.randint(*SHIPPED_RANGES["occ"])
        ypath = POOL / "buildings" / f"{name}.yaml"
        ypath.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        try:
            conf = load_scenario_from_parts(str(ypath), str(KSB / "weather" / f"{weather}.yaml"),
                                            output_base_dir=str(POOL / "scenarios"))
            generate_scenario_from_config(conf)
            return slot, split, name, weather
        except ValueError:  # floor-plan artefact (zone below the kit's minimum area): next seed
            ypath.unlink()
            shutil.rmtree(POOL / "scenarios" / name, ignore_errors=True)
    raise RuntimeError(f"slot {slot}: no valid seed in 100 attempts")


def write_folds() -> None:
    pairs = training_pairs()
    rel = lambda b, w: f"external/starter-kit/smart_buildings/training_scenarios/{b}/{w}"
    folds = {"shipped": [rel(b, w) for b, w in pairs], "loso": {}, "subset10": []}
    for i, st in enumerate(STATIONS):
        test = [rel(b, w) for b, w in pairs if w.startswith(st)]
        folds["loso"][st] = {"test": test, "train": [rel(b, w) for b, w in pairs if not w.startswith(st)]}
        test_sorted = sorted(test)
        folds["subset10"] += [test_sorted[i % 6], test_sorted[(i + 3) % 6]]
    (ROOT / "harness/folds.yaml").write_text(yaml.dump(folds, sort_keys=False))


def load_split(name: str) -> list[Path]:
    """pool_train | pool_test | minneapolis | large | shipped | subset10 | loso:<station>:<test|train>."""
    if name.startswith("loso:"):
        _, st, part = name.split(":")
        rel = yaml.safe_load((ROOT / "harness/folds.yaml").read_text())["loso"][st][part]
    elif name == "pool_train10":  # stratified half of pool_train (2 per station), for tuning
        return load_split("pool_train")[::2]
    elif name == "pool_train10b":  # the other half (2 per station)
        return load_split("pool_train")[1::2]
    elif name in ("shipped", "subset10"):
        rel = yaml.safe_load((ROOT / "harness/folds.yaml").read_text())[name]
    else:
        rel = yaml.safe_load((POOL / "pool.yaml").read_text())[name]
    return [ROOT / r for r in rel]


RL_SLOT0 = 60  # rl_train slots start here: seeds 7000+, disjoint from the evaluation pool's 1000-5999


def add_rl_train(n_per_station: int, workers: int) -> None:
    """Append an `rl_train` split (shipped size ranges, training-set weather strips) without touching any existing
    building: own seeds and own stratified material draws."""
    used = {w for _, w in training_pairs()}
    sl = [("rl_train", w, "shipped") for st in STATIONS for w in [x for x in strips(st) if x in used][:n_per_station]]
    mats = _materials(len(sl), b"aether-rl-v1")
    jobs = [(RL_SLOT0 + i, split, w, kind, mats[i]) for i, (split, w, kind) in enumerate(sl)]
    with Pool(workers) as pool:
        done = sorted(pool.imap_unordered(_make, jobs))
    d = yaml.safe_load((POOL / "pool.yaml").read_text())
    d["rl_train"] = [f"pool/scenarios/{name}/{w}" for _, _, name, w in done]
    (POOL / "pool.yaml").write_text(yaml.dump(d, sort_keys=False))
    print(len(done), "rl_train scenarios")


STRESS_SLOT0 = 100  # stress slots: seeds 11000+


def add_stress(workers: int) -> None:
    """Append a `stress` split: every station (incl. minneapolis) x strips 1/3/5/7 (Jan/Apr/Jul/Oct) x both size
    ranges (shipped, generator default). Own seeds and material draws; existing buildings untouched."""
    sl = [("stress", f"{st}_{k}", kind) for st in STATIONS + ("minneapolis",) for k in (1, 3, 5, 7) for kind in ("shipped", "default")]
    mats = _materials(len(sl), b"aether-stress-v1")
    jobs = [(STRESS_SLOT0 + i, split, w, kind, mats[i]) for i, (split, w, kind) in enumerate(sl)]
    with Pool(workers) as pool:
        done = sorted(pool.imap_unordered(_make, jobs))
    d = yaml.safe_load((POOL / "pool.yaml").read_text())
    d["stress"] = [f"pool/scenarios/{name}/{w}" for _, _, name, w in done]
    (POOL / "pool.yaml").write_text(yaml.dump(d, sort_keys=False))
    print(len(done), "stress scenarios")


RLBIG_SLOT0 = 200  # rl_big slots: seeds 21000+, disjoint from every evaluation split


def add_rl_big(workers: int) -> None:
    """Append `rl_big` for RL training on hidden-like conditions: every station x strips 2/4/6/8 (the months the
    stress grid does not use) x both size ranges. Own seeds and material draws; nothing existing is touched."""
    sl = [("rl_big", f"{st}_{k}", kind) for st in STATIONS + ("minneapolis",) for k in (2, 4, 6, 8) for kind in ("shipped", "default")]
    mats = _materials(len(sl), b"aether-rlbig-v1")
    jobs = [(RLBIG_SLOT0 + i, split, w, kind, mats[i]) for i, (split, w, kind) in enumerate(sl)]
    with Pool(workers) as pool:
        done = sorted(pool.imap_unordered(_make, jobs))
    d = yaml.safe_load((POOL / "pool.yaml").read_text())
    d["rl_big"] = [f"pool/scenarios/{name}/{w}" for _, _, name, w in done]
    (POOL / "pool.yaml").write_text(yaml.dump(d, sort_keys=False))
    print(len(done), "rl_big scenarios")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rl", type=int, default=0, help="only append N rl_train scenarios per station")
    ap.add_argument("--stress", action="store_true", help="only append the stress grid")
    ap.add_argument("--rl_big", action="store_true", help="only append the hidden-like RL training set")
    a = ap.parse_args(argv)
    if a.rl:
        return add_rl_train(a.rl, a.workers)
    if a.stress:
        return add_stress(a.workers)
    if a.rl_big:
        return add_rl_big(a.workers)
    write_folds()
    (POOL / "buildings").mkdir(parents=True, exist_ok=True)
    (POOL / "scenarios").mkdir(parents=True, exist_ok=True)
    sl = slots()
    mats = _materials(len(sl))
    jobs = [(i, split, w, kind, mats[i]) for i, (split, w, kind) in enumerate(sl)]
    with Pool(a.workers) as pool:
        done = sorted(pool.imap_unordered(_make, jobs))
    out: dict[str, list[str]] = {}
    for _, split, name, w in done:
        out.setdefault(split, []).append(f"pool/scenarios/{name}/{w}")
        print(f"{split:<12} {name} {w}")
    (POOL / "pool.yaml").write_text(yaml.dump(out, sort_keys=False))


if __name__ == "__main__":
    main()
