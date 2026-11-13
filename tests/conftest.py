import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "external/starter-kit"
SHIPPED = KIT / "smart_buildings/training_scenarios"


def make_env(scenario_dir: Path):
    """(env, tracker, meta) for a scenario, loaded the way the kit's evaluator does."""
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker

    scenario_dir = Path(scenario_dir).resolve()
    cwd = os.getcwd()
    os.chdir(KIT)
    try:
        env = ev.load_hybrid_environment(scenario_dir / "config.gin")
    finally:
        os.chdir(cwd)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    meta = ev.create_building_meta_data(env, scenario_dir)
    return env, tracker, meta


def first(pattern: str) -> Path:
    return sorted(SHIPPED.glob(pattern))[0]
