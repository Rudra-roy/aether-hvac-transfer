import numpy as np

from tests.test_policy import _snap


class _Parser:
    def __init__(self, K=4, per=2):
        self.groups = [np.arange(k * per, (k + 1) * per) for k in range(K)]
        self.zone_ahu = np.repeat(np.arange(K), per)
        self.ahus = [f"ahu_{k + 1}" for k in range(K)]


def _night(prober, trusted, hours, V=8, K=4, **kw):
    from aether.actions import Command
    from aether.supervisor import SETBACK
    out = []
    for h in hours:
        snap = _snap(V=V, hour=h, dow=2, **kw)
        modes = np.full(V, SETBACK, np.int8)
        cmd, mask = prober.apply(Command.default(V, K), snap, modes, trusted)
        out.append((cmd, mask))
    return out


def _prober(budget=1.0):
    from aether.params import Params
    from aether.probe import Prober
    return Prober(_Parser(), Params(probe_budget_ahu_h=budget), seed=0)


def test_probe_only_unoccupied():
    tr = np.zeros(8, bool)
    assert not _night(_prober(), tr, [2.0], occ=3.0)[0][1].any()
    assert not _night(_prober(), tr, [2.0], soon=True)[0][1].any()
    assert not _night(_prober(), tr, [5.0])[0][1].any()
    assert _night(_prober(), tr, [1.0])[0][1].any()


def test_probe_budget():
    hours = np.arange(0, 5, 1 / 12)
    res = _night(_prober(1.0), np.zeros(8, bool), hours)
    probed = [m for _, m in res if m.any()]
    assert len(probed) == 12
    groups = {tuple(np.flatnonzero(m)) for m in probed}
    assert len(groups) == 1 and len(next(iter(groups))) == 2
    cmd = next(c for c, m in res if m.any())
    k = np.flatnonzero(cmd.ahu_run)
    assert len(k) == 1 and cmd.pump_run


def test_probe_prefers_untrusted():
    tr = np.ones(8, bool)
    tr[4:6] = False  # AHU 3 has no trusted zones
    _, m = _night(_prober(), tr, [1.0])[0]
    assert set(np.flatnonzero(m)) == {4, 5}


def test_probe_excludes_out_of_band():
    from aether.actions import Command
    from aether.supervisor import SETBACK
    p = _prober()
    snap = _snap(V=8, hour=1.0, dow=2)
    snap.T[0] = 287.0  # eco band 289-298: more than 1 K below
    _, m = p.apply(Command.default(8, 4), snap, np.full(8, SETBACK, np.int8), np.zeros(8, bool))
    assert m.any() and not m[0]
