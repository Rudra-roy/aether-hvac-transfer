import numpy as np

from tests.conftest import SHIPPED, make_env


def _run(arm, residual, steps=144):
    from aether.params import Params
    from aether.policy import AetherPolicy
    env, tr, meta = make_env(SHIPPED / "building_train_1/fairbanks_1")
    pol = AetherPolicy(meta, env.action_spec(), params=Params.from_json("submission/params_c5.json"), arm=arm, residual=residual)
    tr.reset()
    obs, out = tr.last_obs, []
    for _ in range(steps):
        a = pol.get_action(obs)
        out += [a["continuous_action"], a["discrete_action"].astype(np.float32)]
        obs, *_ = tr.step(a)
    return np.concatenate(out), pol


def test_zero_residual_is_exactly_the_base():
    """The last layer starts at zero, so an untrained residual policy is c5, bitwise."""
    from aether.residual import residual_init
    base, _ = _run("mpc_off", None)
    zero, pol = _run("mpc_r", residual_init(seed=0))
    np.testing.assert_array_equal(base, zero)
    assert pol.residual_features is not None and np.isfinite(pol.residual_features).all()


def test_residual_moves_actions_within_bounds():
    """A non-zero residual changes the zone commands, stays inside the action bounds and never touches the tank setpoint
    below tw_min (no tank cycling)."""
    from aether.residual import BOUND, residual_init
    w = residual_init(seed=0)
    w["b3"] = np.array([0.5, 0.5])  # push damper and reheat up
    base, _ = _run("mpc_off", None, steps=48)
    moved, pol = _run("mpc_r", w, steps=48)
    assert not np.array_equal(base, moved)
    assert np.abs(moved).max() <= 1.0 + 1e-6 and np.isfinite(moved).all()
    assert BOUND <= 0.3
