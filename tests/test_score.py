import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "external/starter-kit"


def _write_tree(tmp: Path, returns: dict, policy: str) -> None:
    for key, d in returns.items():
        b, w = key.split("/")
        out = tmp / b / w / policy
        out.mkdir(parents=True)
        (out / "summary.json").write_text(json.dumps({"total_reward": d[policy], "total_steps": 8640}))


def test_score_matches_kit(tmp_path):
    from harness.score_table import load_results, score

    returns = json.loads((KIT / "baseline_returns.json").read_text())
    for policy in ("bang_bang", "fixed_setpoint", "pid", "pid_water"):
        tree = tmp_path / policy
        _write_tree(tree, returns, policy)
        res = subprocess.run([sys.executable, str(KIT / "score_leaderboard.py"), str(tree), "--policy", policy],
                             capture_output=True, text=True, check=True)
        kit = float(re.search(r"LEADERBOARD SCORE = ([0-9.]+)", res.stdout).group(1))
        totals = {k: v[policy] for k, v in load_results(tree).items()}
        assert len(totals) == 30
        assert abs(score(totals) - kit) < 5e-6  # kit prints 5 decimals
