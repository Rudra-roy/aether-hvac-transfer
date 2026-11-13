import numpy as np

from tests.test_policy import _snap

C = 273.15


def _ident(theta_d, history=None, V=1):
    from aether.identifier import Prior, ZoneIdentifier
    from aether.zone_model import make_features
    m0 = np.zeros(10)
    m0[9] = theta_d
    ident = ZoneIdentifier(V, Prior(m0=m0, P0=np.full(10, 1e-6), R=1e-4))
    f = make_features(np.zeros(V), np.zeros(V), np.full(V, 285.0), 330.0, 270.0, 0.0, 9.0)
    for T in (history if history is not None else [np.full(V, 21.5 + C)] * 14):
        ident._hist.append((np.asarray(T, float), np.asarray(T, float), f))
    return ident


class _Parser:
    zone_ahu = np.zeros(1, int)
    groups = [np.array([0])]
    n_occ_ref = 2.0


def _apply(ident, T, trusted):
    from aether import guard
    from aether.actions import Command
    from aether.params import Params
    snap = _snap(V=1, hour=9.0, now=True, T=T)
    cmd = Command.default(1, 1)
    return guard.apply(cmd, snap, ident, Params(), _Parser(), np.full(1, 21 + C), np.full(1, 24 + C),
                       watch=np.ones(1, bool), trusted=np.array([trusted]))


def test_guard_fires_on_drop():
    cmd, mask = _apply(_ident(-0.2), 21.5 + C, True)
    assert mask[0] and cmd.reheat[0] == 1.0 and cmd.ahu_run[0] and cmd.pump_run


def test_guard_quiet_in_band():
    cmd, mask = _apply(_ident(0.0, [np.full(1, 22.5 + C)] * 14), 22.5 + C, True)
    assert not mask[0] and cmd.reheat[0] == 0.0


def test_guard_untrusted_trend():
    hist = [np.full(1, 23.0 + C - 0.1 * i) for i in range(14)]  # falling 0.1 K/step, now 21.7 C
    cmd, mask = _apply(_ident(0.0, hist), float(hist[-1][0]), False)
    assert mask[0] and cmd.reheat[0] == 1.0


def test_guard_cools_on_rise():
    cmd, mask = _apply(_ident(+0.2, [np.full(1, 23.5 + C)] * 14), 23.5 + C, True)
    assert mask[0] and cmd.damper[0] == 1.0 and cmd.reheat[0] == 0.0
