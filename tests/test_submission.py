import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import SHIPPED, make_env

ROOT = Path(__file__).resolve().parents[1]


def _scenario():
    return sorted((SHIPPED / "building_train_7").glob("denver_*"))[0]




# The gateway's base class as published in kaggle_evaluation/smart_buildings_inference_server.py (abstract get_action,
# __init__ stores the two arguments). A stand-in, so the test needs neither the competition data nor its gRPC relay.
GATEWAY_STUB = '''
import abc, sys, types
class Policy(abc.ABC):
    def __init__(self, building_meta_data, action_spec):
        self.building_meta_data = building_meta_data
        self.action_spec = action_spec
    @abc.abstractmethod
    def get_action(self, obs): raise NotImplementedError
pkg = types.ModuleType("kaggle_evaluation"); pkg.__path__ = []
srv = types.ModuleType("kaggle_evaluation.smart_buildings_inference_server"); srv.Policy = Policy
pkg.smart_buildings_inference_server = srv
sys.modules["kaggle_evaluation"] = pkg; sys.modules["kaggle_evaluation.smart_buildings_inference_server"] = srv
'''


@pytest.mark.parametrize("arm,params", [("mpc_off", None), ("mpc_v", None), ("mpc_off", "submission/params_zc.json"), ("inv", "submission/params_inv.json"), ("inv", "submission/params_inv2.json"), ("inv", "submission/params_inv3.json")])
def test_notebook_is_plain_code_and_matches_package(tmp_path, arm, params):
    """The Kaggle notebook is the package's modules as plain cells (no embedded sources, no exec, no aether imports),
    and run as Kaggle runs it -- cells in one namespace, gateway base class stubbed -- it acts exactly like the package."""
    import json

    from aether.params import Params
    from aether.policy import AetherPolicy
    from submission.notebook import build

    P = Params.from_json(ROOT / params) if params else Params()
    nb = json.loads(build(tmp_path / "nb.ipynb", params=P, arm=arm).read_text())
    ids = [c.get("id") for c in nb["cells"]]
    assert all(ids) and len(set(ids)) == len(ids), "nbformat 4.5 requires a unique id per cell"
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    src = "\n".join(code)
    for bad in ("_SOURCES", "exec(", "from aether", "import aether", "sys.modules["):
        assert bad not in src, bad
    assert "SmartBuildingsInferenceServer(SubmissionPolicy).run(" in code[-1] and "kaggle_evaluation" in code[0]

    script = GATEWAY_STUB + textwrap.dedent(f"""
        import json, numpy as np
        sys.path.insert(0, {str(ROOT / 'tests')!r})
        from conftest import make_env
        cells = [c for c in json.load(open({str(tmp_path / 'nb.ipynb')!r}))["cells"] if c["cell_type"] == "code"]
        g = {{"__name__": "__main__"}}
        for c in cells[1:-1]:  # everything but the data-mount setup and the server start
            exec("".join(c["source"]), g)
        assert "aether" not in sys.modules
        env, tr, meta = make_env({str(_scenario())!r})
        pol = g["SubmissionPolicy"](meta, env.action_spec())
        assert isinstance(pol, srv.Policy) and pol.building_meta_data is meta
        tr.reset(); obs = tr.last_obs; out = []
        for _ in range(288):
            a = pol.get_action(obs); out += [a["continuous_action"], a["discrete_action"].astype(np.float32)]
            obs, *_ = tr.step(a)
        np.save({str(tmp_path / 'nb.npy')!r}, np.concatenate(out))
    """)
    run = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True, capture_output=True, text=True)
    progress = [l for l in run.stdout.splitlines() if l.startswith("[aether]")]
    assert len(progress) >= 288 // 72 and "ms/step" in progress[-1] and "left" in progress[-1], run.stdout[-500:]

    env, tr, meta = make_env(_scenario())
    pol = AetherPolicy(meta, env.action_spec(), params=P, arm=arm)
    tr.reset()
    obs, out = tr.last_obs, []
    for _ in range(288):
        a = pol.get_action(obs)
        out += [a["continuous_action"], a["discrete_action"].astype(np.float32)]
        obs, *_ = tr.step(a)
    np.testing.assert_array_equal(np.load(tmp_path / "nb.npy"), np.concatenate(out))
